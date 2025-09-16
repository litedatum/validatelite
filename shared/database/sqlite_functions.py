"""
SQLite自定义验证函数

为SQLite提供数值精度验证功能，替代REGEX验证
"""

import re
from typing import Any


def validate_integer_digits(value: Any, max_digits: int) -> bool:
    """
    验证整数位数是否不超过指定位数

    Args:
        value: 待验证的值
        max_digits: 最大允许位数

    Returns:
        bool: True表示验证通过，False表示验证失败

    Examples:
        validate_integer_digits(12345, 5) -> True
        validate_integer_digits(-23456, 5) -> True (负号不算位数)
        validate_integer_digits(123456, 5) -> False
        validate_integer_digits("abc", 5) -> False
        validate_integer_digits(12.34, 5) -> False (有小数部分)
    """
    if value is None:
        return True  # NULL值跳过验证

    try:
        # 尝试转换为浮点数再转换为整数，确保是数值
        float_val = float(value)
        int_val = int(float_val)

        # 检查是否有小数部分
        if float_val != int_val:
            return False  # 有小数部分，不是整数

        # 计算位数（绝对值，去掉负号）
        digit_count = len(str(abs(int_val)))
        return digit_count <= max_digits

    except (ValueError, TypeError, OverflowError):
        return False  # 非法值返回失败


def validate_string_length(value: Any, max_length: int) -> bool:
    """
    验证字符串长度是否不超过指定长度

    Args:
        value: 待验证的值
        max_length: 最大允许长度

    Returns:
        bool: True表示验证通过，False表示验证失败
    """
    if value is None:
        return True  # NULL值跳过验证

    try:
        str_val = str(value)
        return len(str_val) <= max_length
    except Exception:
        return False


def validate_float_precision(value: Any, precision: int, scale: int) -> bool:
    """
    验证浮点数精度和小数位数

    Args:
        value: 待验证的值
        precision: 总精度（整数位+小数位）
        scale: 小数位数

    Returns:
        bool: True表示验证通过，False表示验证失败

    Examples:
        validate_float_precision(123.45, 5, 2) -> True
        validate_float_precision(1234.56, 5, 2) -> False (总位数超过5)
        validate_float_precision(123.456, 5, 2) -> False (小数位超过2)
    """
    if value is None:
        return True  # NULL值跳过验证

    try:
        float_val = float(value)
        val_str = str(float_val)

        # 去掉负号
        if val_str.startswith("-"):
            val_str = val_str[1:]

        if "." in val_str:
            # 有小数点的情况
            integer_part, decimal_part = val_str.split(".")

            # 去掉尾部的0
            decimal_part = decimal_part.rstrip("0")

            # 特殊处理：当precision == scale时，意味着只有小数部分，整数部分必须为0
            if precision == scale:
                # 只允许0.xxxx格式，整数部分必须为0且不计入精度
                if integer_part != "0":
                    return False
                int_digits = 0  # 整数部分的0不计入精度
            else:
                # 正常情况：整数部分计入精度
                int_digits = len(integer_part) if integer_part != "0" else 1

            dec_digits = len(decimal_part)

            # 检查整数位数和小数位数约束
            # 整数位数不能超过 (precision - scale)，小数位数不能超过 scale
            max_integer_digits = precision - scale
            return int_digits <= max_integer_digits and dec_digits <= scale
        else:
            # 整数情况
            int_digits = len(val_str) if val_str != "0" else 1
            # 整数也要遵守precision-scale约束
            max_integer_digits = precision - scale
            return int_digits <= max_integer_digits

    except (ValueError, TypeError, OverflowError):
        return False


def validate_integer_range_by_digits(value: Any, max_digits: int) -> bool:
    """
    通过范围检查来验证整数位数（备用方案）

    Args:
        value: 待验证的值
        max_digits: 最大允许位数

    Returns:
        bool: True表示验证通过，False表示验证失败
    """
    if value is None:
        return True

    try:
        int_val = int(float(value))
        max_val = 10**max_digits - 1  # 例如：5位数的最大值是99999
        min_val = -(10**max_digits - 1)  # 例如：5位数的最小值是-99999
        return min_val <= int_val <= max_val
    except (ValueError, TypeError, OverflowError):
        return False


# 为了方便SQLite注册，提供失败检测版本
def detect_invalid_integer_digits(value: Any, max_digits: int) -> bool:
    """检测不符合整数位数要求的值（用于COUNT失败记录）"""
    return not validate_integer_digits(value, max_digits)


def detect_invalid_string_length(value: Any, max_length: int) -> bool:
    """检测不符合字符串长度要求的值"""
    return not validate_string_length(value, max_length)


def detect_invalid_float_precision(value: Any, precision: int, scale: int) -> bool:
    """检测不符合浮点数精度要求的值"""
    return not validate_float_precision(value, precision, scale)
