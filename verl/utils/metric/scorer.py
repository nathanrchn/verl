"""
AutoScoringJudge for evaluating reasoning tasks.
Adapted from reasoning_evals/checker.py

Supports multiple answer types:
- Exact string match
- Multiple choice (A, B, C, D, E)
- Numerical equality with tolerance
- Mathematical expression equality (requires sympy)
- Interval equality
- Equation equality
"""
import re
import math
from typing import Optional

# Lazy imports for optional dependencies
_sympy = None
_parse_latex = None


def _init_sympy():
    """Lazily initialize sympy to avoid import errors if not needed."""
    global _sympy, _parse_latex
    if _sympy is None:
        try:
            import sympy as sp
            from sympy.parsing.latex import parse_latex
            _sympy = sp
            _parse_latex = parse_latex
        except ImportError:
            pass
    return _sympy is not None


class AutoScoringJudge:
    """
    Multi-strategy scoring judge for evaluating model outputs against ground truth.

    Supports:
    - Exact match
    - MCQ matching (A, B, C, D, E)
    - Numerical equality with precision tolerance
    - Mathematical expression equality (LaTeX)
    - Interval equality
    - Equation equality
    """

    def __init__(self, precision: float = 1e-8):
        self.special_signal_map = {
            "\\left": "",
            "\\right": "",
            "∶": ":",
            "，": ",",
            "$": "",
            "\\approx": "=",
            "\\simeq": "=",
            "\\sim": "=",
            "^\\prime": "'",
            "^{\\prime}": "'",
            "^\\circ": "",
            "%": "",
        }
        self.precision = precision
        self._pi = None

    @property
    def pi(self):
        """Lazily initialize pi symbol."""
        if self._pi is None and _init_sympy():
            # Use sympy.pi directly instead of parsing LaTeX
            self._pi = _sympy.pi
        return self._pi

    def split_by_comma(self, expr: str) -> list[str]:
        """Split expressions by commas outside of brackets."""
        in_bracket_num = 0
        splitted_expr = []
        start_idx = 0
        for i, char in enumerate(expr):
            if char in ["(", "["]:
                in_bracket_num += 1
            elif char in [")", "]"]:
                in_bracket_num -= 1
            elif char == "," and in_bracket_num == 0:
                splitted_expr.append(expr[start_idx:i].strip())
                start_idx = i + 1

        if start_idx < len(expr):
            splitted_expr.append(expr[start_idx:].strip())

        return splitted_expr

    def trans_plus_minus_sign(self, expr_list: list[str]) -> list[str]:
        """Translate plus-minus signs into separate expressions."""
        new_expr_list = []
        for expr in expr_list:
            if "\\pm" in expr:
                new_expr_list.append(expr.replace("\\pm", "+"))
                new_expr_list.append(expr.replace("\\pm", "-"))
            else:
                new_expr_list.append(expr)
        return new_expr_list

    def judge(self, ground_truth: str, prediction: str, precision: Optional[float] = None) -> bool:
        """
        Judge if prediction matches ground truth.

        Args:
            ground_truth: The expected answer
            prediction: The model's prediction
            precision: Optional precision override for numerical comparisons

        Returns:
            True if prediction matches ground truth, False otherwise
        """
        if precision is None:
            precision = self.precision
        precision_list = [precision] if not isinstance(precision, list) else precision

        try:
            expression1, expression2 = self.preprocess(ground_truth, prediction)
        except Exception:
            return False

        if expression1 == expression2:
            return True

        # Check for multiple choice answer equality
        try:
            if self.mcq_equal(expression1, expression2):
                return True
        except Exception:
            pass

        # Remove Chinese characters
        expression1 = re.sub(r'[\u4e00-\u9fff]+', '', expression1)
        expression2 = re.sub(r'[\u4e00-\u9fff]+', '', expression2)

        expression1_list = self.split_by_comma(expression1)
        expression2_list = self.split_by_comma(expression2)

        temp_list1 = self.trans_plus_minus_sign(expression1_list)
        temp_list2 = self.trans_plus_minus_sign(expression2_list)

        # Extend precision list if needed
        if len(precision_list) <= 1:
            precision_list = precision_list * len(temp_list1)

        if len(temp_list1) != len(temp_list2):
            return False

        # Check if elements in both lists can be paired and are equal
        idx = -1
        while len(temp_list1) != 0:
            idx = (idx + 1) % len(temp_list1)
            item1 = temp_list1[idx]
            current_precision = precision_list[idx]

            for item2 in temp_list2:
                if self.is_equal(item1, item2, current_precision):
                    temp_list1.remove(item1)
                    temp_list2.remove(item2)
                    precision_list.remove(current_precision)
                    break
            else:
                return False

        return True

    def is_interval(self, expr: str) -> bool:
        """Check if an expression is an interval."""
        return expr.startswith(("(", "[")) and expr.endswith((")", "]"))

    def sympy_sub_pi(self, expression_sympy):
        """Replace the symbol for pi in sympy expressions with its numerical value."""
        if self.pi is not None and _sympy is not None:
            return expression_sympy.subs(self.pi, math.pi)
        return expression_sympy

    def is_equal(self, expression1: str, expression2: str, precision: float) -> bool:
        """Check if two expressions are equal using multiple strategies."""
        if expression1 == expression2 and expression1 != "" and expression2 != "":
            return True

        # Check for interval equality
        if self.is_interval(expression1) and self.is_interval(expression2):
            try:
                if self.interval_equal(expression1, expression2, precision):
                    return True
            except Exception:
                return False

        # Check for numerical equality
        try:
            if self.numerical_equal(expression1, expression2, precision):
                return True
        except Exception:
            pass

        # Check for mathematical expression equality (requires sympy)
        if _init_sympy():
            try:
                if self.expression_equal(expression1, expression2, precision) and \
                   not ("=" in expression1 and "=" in expression2):
                    return True
            except Exception:
                pass
            try:
                if self.equation_equal(expression1, expression2):
                    return True
            except Exception:
                pass

        # Check for numerical with unit equality
        try:
            if self.numerical_with_unit_equal(expression1, expression2, precision):
                return True
        except Exception:
            pass

        return False

    def mcq_equal(self, expression1: str, expression2: str) -> bool:
        """Check for MCQ answer equality (A, B, C, D, E)."""
        if expression1.upper() in ["A", "B", "C", "D", "E"]:
            # Match patterns like "A. explanation" or "A) explanation"
            if re.match(r"[ABCDEabcde][)\.]*\s.*", expression2) and \
               expression2[0].upper() == expression1.upper():
                return True
            # Also match just the letter
            if expression2.upper().strip() == expression1.upper():
                return True
        return False

    def numerical_with_unit_equal(self, expression1: str, expression2: str, precision: float) -> bool:
        """Check for numerical equality when answer has units."""
        match = re.findall(r"^-?\d*\.?\d+", expression2)
        if match:
            expression2_num = match[0]
            return self.numerical_equal(expression1, expression2_num, precision)
        return False

    def numerical_equal(self, expression1: str, expression2: str, precision: float,
                       include_percentage: bool = False) -> bool:
        """Check if two numerical values are equal within precision."""
        reference = float(expression1)
        prediction = float(expression2)

        if include_percentage:
            gt_results = [reference / 100, reference, reference * 100]
        else:
            gt_results = [reference]

        for item in gt_results:
            if abs(item - prediction) <= precision * 1.01:
                return True
        return False

    def _parse_expression(self, expr: str):
        """
        Parse an expression into a sympy object.

        Tries sympify first for plain expressions, then parse_latex for LaTeX.
        This order prioritizes simpler expressions and avoids antlr4 dependency issues.
        """
        if not _init_sympy():
            return None

        # Try sympify first for plain expressions (e.g., "1/2", "0.5", "x+1")
        # This handles most common cases without needing LaTeX parsing
        try:
            result = _sympy.sympify(expr, evaluate=True)
            if result is not None:
                return result
        except Exception:
            pass

        # Fall back to parse_latex for LaTeX-specific syntax (e.g., \sqrt, \pi)
        if _parse_latex is not None:
            try:
                return _sympy.sympify(_parse_latex(expr))
            except Exception:
                pass

        return None

    def expression_equal(self, exp1: str, exp2: str, precision: float) -> bool:
        """Check if two expressions are mathematically equivalent using sympy."""
        if not _init_sympy():
            return False

        def extract_expression(expression: str) -> str:
            if "=" in expression:
                expression = expression.split("=")[1]
            return expression.strip()

        exp1 = extract_expression(exp1)
        exp2 = extract_expression(exp2)

        expr1_sym = self._parse_expression(exp1)
        expr2_sym = self._parse_expression(exp2)

        if expr1_sym is None or expr2_sym is None:
            return False

        if expr1_sym == expr2_sym:
            return True

        expr1_sym = self.sympy_sub_pi(expr1_sym)
        expr2_sym = self.sympy_sub_pi(expr2_sym)

        if (expr1_sym.has(_sympy.Symbol) and not expr2_sym.has(_sympy.Symbol)) or \
           (not expr1_sym.has(_sympy.Symbol) and expr2_sym.has(_sympy.Symbol)):
            return False
        elif not expr1_sym.has(_sympy.Symbol) and not expr2_sym.has(_sympy.Symbol):
            try:
                if not (self.can_compute_power(expr1_sym) and self.can_compute_power(expr2_sym)):
                    return False
                if abs(expr1_sym.evalf() - expr2_sym.evalf()) <= precision * 1.01:
                    return True
                return False
            except Exception:
                return False
        else:
            try:
                simplified_expr = _sympy.simplify(expr1_sym - expr2_sym)
                num_value = simplified_expr.evalf()
                return abs(num_value) < 1e-3
            except Exception:
                return False

    def equation_equal(self, expression1: str, expression2: str) -> bool:
        """Check if two equations are mathematically equivalent."""
        if not _init_sympy():
            return False

        def simplify_equation(latex_eq: str):
            lhs, rhs = latex_eq.split('=')
            lhs_expr = self._parse_expression(lhs)
            rhs_expr = self._parse_expression(rhs)
            if lhs_expr is None or rhs_expr is None:
                return None
            equation = _sympy.Eq(lhs_expr, rhs_expr)
            return _sympy.simplify(equation.lhs - equation.rhs)

        expr1_sym = simplify_equation(expression1)
        expr2_sym = simplify_equation(expression2)

        if expr1_sym is None or expr2_sym is None:
            return False

        division_result_1 = _sympy.simplify(expr1_sym / expr2_sym)
        division_result_2 = _sympy.simplify(expr2_sym / expr1_sym)

        if (division_result_1.is_Integer and division_result_1 != 0) or \
           (division_result_2.is_Integer and division_result_2 != 0):
            return True
        return False

    def interval_equal(self, expression1: str, expression2: str, precision: float) -> bool:
        """Check if two intervals are mathematically equivalent."""
        def compare_two_interval(inter1: str, inter2: str) -> bool:
            if inter1[0] != inter2[0] or inter1[-1] != inter2[-1]:
                return False

            inter1 = inter1.strip('[]()')
            inter2 = inter2.strip('[]()')

            items_1 = inter1.split(',')
            items_2 = inter2.split(',')

            for item_1, item_2 in zip(items_1, items_2):
                if not self.is_equal(item_1, item_2, precision):
                    return False
            return True

        if expression1 == expression2:
            return True

        inter_list1 = expression1.split("\\cup")
        inter_list2 = expression2.split("\\cup")

        if len(inter_list1) != len(inter_list2):
            return False

        for inter1, inter2 in zip(inter_list1, inter_list2):
            if not compare_two_interval(inter1, inter2):
                return False
        return True

    def normalize_fraction(self, expr: str) -> str:
        """
        Normalize fraction representations.

        Converts:
        - \\frac{a}{b} -> a/b
        - \\dfrac{a}{b} -> a/b
        - \\tfrac{a}{b} -> a/b

        Handles nested braces in numerator and denominator.
        """
        def extract_brace_content(s: str, start: int) -> tuple[str, int]:
            """Extract content within braces, handling nesting."""
            if start >= len(s) or s[start] != '{':
                return "", start

            depth = 1
            end = start + 1
            while end < len(s) and depth > 0:
                if s[end] == '{':
                    depth += 1
                elif s[end] == '}':
                    depth -= 1
                end += 1

            return s[start + 1:end - 1], end

        result = expr
        # Match \frac, \dfrac, \tfrac
        frac_pattern = r'\\[dt]?frac\s*{'

        while True:
            match = re.search(frac_pattern, result)
            if not match:
                break

            # Find the start of the first brace
            brace_start = match.end() - 1
            numerator, after_num = extract_brace_content(result, brace_start)

            # Skip whitespace between numerator and denominator
            while after_num < len(result) and result[after_num] in ' \t\n':
                after_num += 1

            denominator, after_denom = extract_brace_content(result, after_num)

            if numerator and denominator:
                # Recursively normalize any nested fractions
                numerator = self.normalize_fraction(numerator)
                denominator = self.normalize_fraction(denominator)

                # Replace the entire \frac{...}{...} with (numerator)/(denominator)
                # Add parentheses only if needed for complex expressions
                if len(numerator) > 1 and not numerator.isdigit():
                    num_str = f"({numerator})"
                else:
                    num_str = numerator
                if len(denominator) > 1 and not denominator.isdigit():
                    denom_str = f"({denominator})"
                else:
                    denom_str = denominator

                replacement = f"{num_str}/{denom_str}"
                result = result[:match.start()] + replacement + result[after_denom:]
            else:
                break

        return result

    def preprocess(self, expression1: str, expression2: str) -> tuple[str, str]:
        """Preprocess expressions to extract and replace special symbols."""
        def extract_boxed_content(latex_str: str) -> str:
            boxed_matches = re.finditer(r'\\boxed{', latex_str)
            results = ""

            for match in boxed_matches:
                start_index = match.end()
                end_index = start_index
                stack = 1

                while stack > 0 and end_index < len(latex_str):
                    if latex_str[end_index] == '{':
                        stack += 1
                    elif latex_str[end_index] == '}':
                        stack -= 1
                    end_index += 1

                if stack == 0:
                    content = latex_str[start_index:end_index - 1]
                    results += content + ","
                else:
                    raise ValueError("Mismatched braces in LaTeX string.")

            if results == "":
                last_line_ans = latex_str.strip().split("\n")[-1]
                dollar_pattern = r"\$(.*?)\$"
                answers = re.findall(dollar_pattern, last_line_ans)

                if answers:
                    for ans in answers:
                        results += ans + ","
                else:
                    results = latex_str

            return results

        def special_symbol_replace(expression: str) -> str:
            if "\\in " in expression:
                expression = expression.split("\\in ")[1]

            for signal, replacement in self.special_signal_map.items():
                expression = expression.replace(signal, replacement)

            expression = expression.strip("\n$,.:;^_=+`!@#$%^&*~，。")

            pattern = r'\\(?:mathrm|mathbf)\{~?([^}]*)\}'
            expression = re.sub(pattern, r'\1', expression)

            return expression

        exp1 = extract_boxed_content(expression1)
        exp2 = extract_boxed_content(expression2)
        exp1 = special_symbol_replace(exp1)
        exp2 = special_symbol_replace(exp2)

        # Normalize fractions to a common format (a/b)
        exp1 = self.normalize_fraction(exp1)
        exp2 = self.normalize_fraction(exp2)

        return exp1, exp2

    def can_compute_power(self, expr) -> bool:
        """Check if a power expression can be computed."""
        if not _init_sympy():
            return True
        if isinstance(expr, _sympy.Pow):
            base, exp = expr.as_base_exp()
            if base.is_number and exp.is_number:
                MAX_EXP = 1000
                if abs(exp.evalf()) > MAX_EXP:
                    return False
        return True


# Global singleton instance
_scorer_instance: Optional[AutoScoringJudge] = None


def get_scorer() -> AutoScoringJudge:
    """Get the global scorer instance."""
    global _scorer_instance
    if _scorer_instance is None:
        _scorer_instance = AutoScoringJudge()
    return _scorer_instance
