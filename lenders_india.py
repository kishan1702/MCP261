"""Public and private lenders list in India."""

PUBLIC_SECTOR_LENDERS = [
    "Bank of Baroda",
    "Bank of India",
    "Bank of Maharashtra",
    "Canara Bank",
    "Central Bank of India",
    "Indian Bank",
    "Indian Overseas Bank",
    "Punjab & Sind Bank",
    "Punjab National Bank",
    "State Bank of India",
    "UCO Bank",
    "Union Bank of India",
]

PRIVATE_SECTOR_LENDERS = [
    "Axis Bank",
    "Bandhan Bank",
    "CSB Bank",
    "City Union Bank",
    "DCB Bank",
    "Dhanlaxmi Bank",
    "Federal Bank",
    "HDFC Bank",
    "ICICI Bank",
    "IDBI Bank",
    "IDFC FIRST Bank",
    "IndusInd Bank",
    "Jammu & Kashmir Bank",
    "Karnataka Bank",
    "Karur Vysya Bank",
    "Kotak Mahindra Bank",
    "Nainital Bank",
    "RBL Bank",
    "South Indian Bank",
    "Tamilnad Mercantile Bank",
    "YES Bank",
]


def get_lenders_list():
    """Return lenders grouped by public and private ownership."""
    return {
        "public": PUBLIC_SECTOR_LENDERS,
        "private": PRIVATE_SECTOR_LENDERS,
    }


def format_lenders_list():
    """Return a readable lenders list."""
    lenders = get_lenders_list()
    public_lines = "\n".join(f"- {name}" for name in lenders["public"])
    private_lines = "\n".join(f"- {name}" for name in lenders["private"])
    return (
        "Public lenders in India:\n"
        f"{public_lines}\n\n"
        "Private lenders in India:\n"
        f"{private_lines}"
    )


if __name__ == "__main__":
    print(format_lenders_list())
