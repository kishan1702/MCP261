import unittest

from lenders_india import get_lenders_list, format_lenders_list


class LendersIndiaTest(unittest.TestCase):
    def test_grouped_lenders_contains_public_and_private(self):
        lenders = get_lenders_list()

        self.assertIn("public", lenders)
        self.assertIn("private", lenders)
        self.assertGreater(len(lenders["public"]), 0)
        self.assertGreater(len(lenders["private"]), 0)

    def test_formatted_output_contains_headings(self):
        output = format_lenders_list()

        self.assertIn("Public lenders in India:", output)
        self.assertIn("Private lenders in India:", output)
        self.assertIn("State Bank of India", output)
        self.assertIn("HDFC Bank", output)


if __name__ == "__main__":
    unittest.main()
