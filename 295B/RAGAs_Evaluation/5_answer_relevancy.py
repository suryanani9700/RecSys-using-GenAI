from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric
import os

# Get the OpenAI API key from the environment
openai_key = os.getenv("OPENAI_API_KEY")
if openai_key is None:
    raise ValueError("OpenAI API key is missing. Please set it using 'export OPENAI_API_KEY=your-openai-key' in the terminal.")
else:
    os.environ["OPENAI_API_KEY"] = openai_key
    print("OpenAI API key set successfully.")

def test_answer_relevancy():
    answer_relevancy_metric = AnswerRelevancyMetric(threshold=0.5)
    test_case = LLMTestCase(
        input="chicken biryani",
        actual_output="""
        Recommended Shopping List:

Based on the provided context, I would recommend the following 5 items from the ""title"" column to cook chicken biryani:

Rao's Homemade Margherita Pizza Sauce (from the Grocery category) - While it's a tomato sauce, you could use it as a base for your biryani marinade or as an ingredient in the rice mixture.
Patons Macadamia Roasted with Sea Salt Macadamia Nuts (from the Grocery category) - The roasted nuts can add a nice crunch and flavor to your chicken biryani.
Party Partners Design 12 Count Santa Party Picks (from the Amazon Home category) - Wait, this one doesn't seem related... or does it? You could use some of the garlands or decorations as a wrapping material for the dish or as a garnish.
To complete the list, I would recommend two more items from different categories:

Lane's BBQ Rub (from the Grocery category) - While it's a BBQ seasoning, you can also use it to add flavor to your chicken biryani.
Party Starter (Small Steak Combo Pack) (from the Grocery category) - The combo pack contains various steak seasonings and sauces that might be useful in preparing the chicken biryani.
Please note that these recommendations are quite unconventional, as they're not typically used in traditional chicken biryani recipes. However, feel free to get creative and experiment with different ingredients!
        """,
        expected_output="""
        "
Royal Basmati Rice, 15-Pound Bag
Shan Biryani Masala Spice Mix, 50 Grams (1.75 Ounces)
Maya Kaimal Indian Simmer Sauce, Kashmiri Curry, Mild, 12.5 oz
Tasty Bite Brown Basmati Rice, 8.8 Ounce (Pack of 6), Ready to Eat, Fully Cooked, Aged Basmati Rice
Swad Biryani Masala, 7-Ounce Boxes (Pack of 6)"
        """,
        context=None
    )
    assert_test(test_case, [answer_relevancy_metric])
