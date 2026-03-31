"""
환율 변환 도구
전 세계 주요 통화 간 변환 (시뮬레이션 데이터)
"""

from langchain_core.tools import tool


# USD 기준 고정 환율 (시뮬레이션)
_RATES_TO_USD: dict[str, float] = {
    "USD": 1.0,
    "KRW": 0.00075,    # 1 USD ≈ 1,333 KRW
    "TRY": 0.031,      # 1 USD ≈ 32.3 TRY
    "EUR": 1.08,        # 1 EUR = 1.08 USD
    "JPY": 0.0067,      # 1 USD ≈ 149 JPY
    "GBP": 1.26,        # British Pound
    "CNY": 0.14,        # Chinese Yuan
    "THB": 0.028,       # Thai Baht
    "VND": 0.000041,    # Vietnamese Dong
    "PHP": 0.018,       # Philippine Peso
    "SGD": 0.74,        # Singapore Dollar
    "AUD": 0.65,        # Australian Dollar
    "CAD": 0.74,        # Canadian Dollar
    "CHF": 1.12,        # Swiss Franc
    "INR": 0.012,       # Indian Rupee
    "MXN": 0.058,       # Mexican Peso
    "BRL": 0.20,        # Brazilian Real
    "AED": 0.27,        # UAE Dirham
    "SEK": 0.096,       # Swedish Krona
    "NZD": 0.61,        # New Zealand Dollar
    "IDR": 0.000063,    # Indonesian Rupiah
    "MYR": 0.22,        # Malaysian Ringgit
    "EGP": 0.020,       # Egyptian Pound
    "MAD": 0.10,        # Moroccan Dirham
    "ZAR": 0.055,       # South African Rand
}

# 통화 기호 매핑
CURRENCY_SYMBOLS: dict[str, str] = {
    "USD": "$", "KRW": "₩", "TRY": "₺", "EUR": "€", "JPY": "¥",
    "GBP": "£", "CNY": "¥", "THB": "฿", "VND": "₫", "PHP": "₱",
    "SGD": "S$", "AUD": "A$", "CAD": "C$", "CHF": "CHF", "INR": "₹",
    "MXN": "MX$", "BRL": "R$", "AED": "AED", "SEK": "kr", "NZD": "NZ$",
    "IDR": "Rp", "MYR": "RM", "EGP": "E£", "MAD": "MAD", "ZAR": "R",
}

SUPPORTED_CURRENCIES = ", ".join(sorted(_RATES_TO_USD.keys()))


@tool
def convert_currency(amount: float, from_currency: str, to_currency: str) -> str:
    """Convert between world currencies.
    Supported: USD, EUR, GBP, JPY, KRW, TRY, CNY, THB, SGD, AUD, CAD, CHF, INR, and more.
    Example: convert_currency(100, 'USD', 'EUR')"""
    try:
        from_cur = from_currency.strip().upper()
        to_cur = to_currency.strip().upper()

        if from_cur not in _RATES_TO_USD:
            return f"'{from_currency}' is not supported.\nSupported: {SUPPORTED_CURRENCIES}"
        if to_cur not in _RATES_TO_USD:
            return f"'{to_currency}' is not supported.\nSupported: {SUPPORTED_CURRENCIES}"
        if amount <= 0:
            return "Amount must be greater than 0."

        usd_amount = amount * _RATES_TO_USD[from_cur]
        converted = usd_amount / _RATES_TO_USD[to_cur]

        from_symbol = CURRENCY_SYMBOLS.get(from_cur, from_cur)
        to_symbol = CURRENCY_SYMBOLS.get(to_cur, to_cur)

        # 소수점 자릿수 결정
        if to_cur in ("KRW", "JPY", "VND", "IDR"):
            converted_str = f"{converted:,.0f}"
        else:
            converted_str = f"{converted:,.2f}"

        return (
            f"💱 Currency conversion:\n"
            f"  {from_symbol}{amount:,.2f} {from_cur} = {to_symbol}{converted_str} {to_cur}\n"
            f"  (Rate: 1 {from_cur} = {_RATES_TO_USD[from_cur] / _RATES_TO_USD[to_cur]:.6f} {to_cur})"
        )

    except Exception as e:
        return f"Currency conversion error: {str(e)}"
