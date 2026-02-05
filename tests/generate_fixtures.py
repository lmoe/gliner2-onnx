#!/usr/bin/env python3
"""
Generate test fixtures by running test cases through native GLiNER and GLiNER2.

This creates JSON files that serve as ground truth for testing
both Python and Node.js ONNX runtimes.

Usage:
    uv run python tests/generate_fixtures.py

Output:
    tests/gliner1.fixtures.json - NER fixtures for GLiNER1 ONNX models
    tests/gliner2.fixtures.json - Classification and NER fixtures for GLiNER2
"""

import json
import logging
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gliner import GLiNER
    from gliner2 import GLiNER2

# Suppress warnings before imports
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("gliner").setLevel(logging.ERROR)
logging.getLogger("gliner2").setLevel(logging.ERROR)

# fmt: off
CLASSIFICATION_TESTS = [
    # Sentiment
    ("This movie was absolutely fantastic and I loved every minute!", ["positive", "negative", "neutral"]),
    ("The food was terrible and the service was slow.", ["positive", "negative", "neutral"]),
    ("It was okay, nothing special.", ["positive", "negative", "neutral"]),
    ("I'm so happy with my purchase!", ["positive", "negative", "neutral"]),
    ("Worst experience ever, never going back.", ["positive", "negative", "neutral"]),
    ("The product works as expected.", ["positive", "negative", "neutral"]),
    ("Absolutely disgusting, I want a refund.", ["positive", "negative", "neutral"]),
    ("Best decision I ever made!", ["positive", "negative", "neutral"]),
    ("Meh, it's fine I guess.", ["positive", "negative", "neutral"]),
    ("Outstanding quality and fast shipping!", ["positive", "negative", "neutral"]),

    # Shopping vs Work vs Entertainment
    ("I need to buy groceries for dinner tonight.", ["shopping", "work", "entertainment"]),
    ("The quarterly report is due tomorrow.", ["shopping", "work", "entertainment"]),
    ("Let's watch Netflix and chill.", ["shopping", "work", "entertainment"]),
    ("Add milk and eggs to the shopping list.", ["shopping", "work", "entertainment"]),
    ("Meeting with the client at 3pm.", ["shopping", "work", "entertainment"]),
    ("Going to the cinema to see the new Marvel movie.", ["shopping", "work", "entertainment"]),
    ("Order new office supplies from Amazon.", ["shopping", "work", "entertainment"]),
    ("Finish the presentation slides.", ["shopping", "work", "entertainment"]),
    ("Playing video games all weekend.", ["shopping", "work", "entertainment"]),
    ("Buy a birthday present for mom.", ["shopping", "work", "entertainment"]),

    # Medical vs Travel vs Finance
    ("Doctor appointment tomorrow at 2pm.", ["medical", "travel", "finance"]),
    ("Flight to Paris leaves Monday morning.", ["medical", "travel", "finance"]),
    ("Pay the credit card bill before Friday.", ["medical", "travel", "finance"]),
    ("Need to refill my prescription.", ["medical", "travel", "finance"]),
    ("Book hotel in Barcelona for next week.", ["medical", "travel", "finance"]),
    ("Check my investment portfolio.", ["medical", "travel", "finance"]),
    ("Annual checkup with Dr. Smith.", ["medical", "travel", "finance"]),
    ("Rent a car at the airport.", ["medical", "travel", "finance"]),
    ("Transfer money to savings account.", ["medical", "travel", "finance"]),
    ("Get blood test results from the lab.", ["medical", "travel", "finance"]),

    # Food vs Fitness vs Social
    ("Try the new Italian restaurant downtown.", ["food", "fitness", "social"]),
    ("Go to the gym after work.", ["food", "fitness", "social"]),
    ("Meet friends for drinks on Saturday.", ["food", "fitness", "social"]),
    ("Order pizza for dinner.", ["food", "fitness", "social"]),
    ("Run 5 kilometers in the morning.", ["food", "fitness", "social"]),
    ("Birthday party at Sarah's house.", ["food", "fitness", "social"]),
    ("Cook pasta with homemade sauce.", ["food", "fitness", "social"]),
    ("Yoga class at 6am.", ["food", "fitness", "social"]),
    ("Game night with the neighbors.", ["food", "fitness", "social"]),
    ("Bake cookies for the kids.", ["food", "fitness", "social"]),

    # Technology vs Education vs Home
    ("Update the software on my laptop.", ["technology", "education", "home"]),
    ("Study for the exam next week.", ["technology", "education", "home"]),
    ("Fix the leaky faucet in the bathroom.", ["technology", "education", "home"]),
    ("Install the new app on my phone.", ["technology", "education", "home"]),
    ("Read chapter 5 of the textbook.", ["technology", "education", "home"]),
    ("Paint the bedroom walls.", ["technology", "education", "home"]),
    ("Configure the new router.", ["technology", "education", "home"]),
    ("Finish the online course.", ["technology", "education", "home"]),
    ("Clean the garage.", ["technology", "education", "home"]),
    ("Back up all my photos to the cloud.", ["technology", "education", "home"]),

    # Urgent vs Routine vs Optional
    ("Emergency! Call 911 immediately!", ["urgent", "routine", "optional"]),
    ("Weekly grocery shopping.", ["urgent", "routine", "optional"]),
    ("Maybe try that new coffee shop.", ["urgent", "routine", "optional"]),
    ("Server is down, fix now!", ["urgent", "routine", "optional"]),
    ("Daily standup meeting.", ["urgent", "routine", "optional"]),
    ("Could reorganize the bookshelf someday.", ["urgent", "routine", "optional"]),
    ("Critical bug in production!", ["urgent", "routine", "optional"]),
    ("Monthly report submission.", ["urgent", "routine", "optional"]),
    ("Might watch that documentary later.", ["urgent", "routine", "optional"]),
    ("Patient needs immediate attention!", ["urgent", "routine", "optional"]),

    # Business categories
    ("Schedule interview with candidate.", ["hiring", "sales", "support"]),
    ("Close the deal with the new client.", ["hiring", "sales", "support"]),
    ("Customer complaint about billing.", ["hiring", "sales", "support"]),
    ("Review resumes for the position.", ["hiring", "sales", "support"]),
    ("Prepare sales pitch for tomorrow.", ["hiring", "sales", "support"]),
    ("Resolve ticket #12345.", ["hiring", "sales", "support"]),
    ("Onboard new employee.", ["hiring", "sales", "support"]),
    ("Follow up with prospect.", ["hiring", "sales", "support"]),
    ("User can't login to their account.", ["hiring", "sales", "support"]),
    ("Reference check for applicant.", ["hiring", "sales", "support"]),

    # Communication intent
    ("Please send me the document.", ["request", "inform", "question"]),
    ("The meeting has been moved to 4pm.", ["request", "inform", "question"]),
    ("What time does the store close?", ["request", "inform", "question"]),
    ("Can you help me with this task?", ["request", "inform", "question"]),
    ("FYI: The project is complete.", ["request", "inform", "question"]),
    ("How do I reset my password?", ["request", "inform", "question"]),
    ("I need access to the shared drive.", ["request", "inform", "question"]),
    ("Update: Sales are up 20%.", ["request", "inform", "question"]),
    ("Where is the nearest ATM?", ["request", "inform", "question"]),
    ("Please approve my vacation request.", ["request", "inform", "question"]),

    # Weather/Nature
    ("It's raining heavily outside.", ["weather", "nature", "disaster"]),
    ("Beautiful sunset over the mountains.", ["weather", "nature", "disaster"]),
    ("Earthquake warning issued.", ["weather", "nature", "disaster"]),
    ("Sunny and warm today.", ["weather", "nature", "disaster"]),
    ("Spotted a deer in the backyard.", ["weather", "nature", "disaster"]),
    ("Hurricane approaching the coast.", ["weather", "nature", "disaster"]),
    ("Snow forecast for tomorrow.", ["weather", "nature", "disaster"]),
    ("The flowers are blooming.", ["weather", "nature", "disaster"]),
    ("Wildfire spreading rapidly.", ["weather", "nature", "disaster"]),
    ("Perfect weather for a picnic.", ["weather", "nature", "disaster"]),

    # Sports vs Music vs Art (additional 10 to reach 100)
    ("The basketball game starts at 7pm.", ["sports", "music", "art"]),
    ("Listen to the new album by Taylor Swift.", ["sports", "music", "art"]),
    ("Visit the modern art museum downtown.", ["sports", "music", "art"]),
    ("Watch the Super Bowl on Sunday.", ["sports", "music", "art"]),
    ("Learn to play guitar this summer.", ["sports", "music", "art"]),
    ("The painting exhibition opens tomorrow.", ["sports", "music", "art"]),
    ("Join the local soccer league.", ["sports", "music", "art"]),
    ("Concert tickets for the rock festival.", ["sports", "music", "art"]),
    ("Take a photography class.", ["sports", "music", "art"]),
    ("Marathon training starts next month.", ["sports", "music", "art"]),
]

NER_TESTS = [
    # People and Organizations
    ("John Smith works at Microsoft in Seattle.", ["person", "organization", "location"]),
    ("Elon Musk announced new Tesla features.", ["person", "organization"]),
    ("Dr. Sarah Johnson joined Harvard University.", ["person", "organization"]),
    ("Tim Cook presented at Apple's keynote.", ["person", "organization"]),
    ("Mark Zuckerberg founded Facebook in 2004.", ["person", "organization"]),
    ("Jeff Bezos stepped down from Amazon.", ["person", "organization"]),
    ("Sundar Pichai leads Google.", ["person", "organization"]),
    ("Warren Buffett invested in Coca-Cola.", ["person", "organization"]),
    ("Bill Gates donated to the WHO.", ["person", "organization"]),
    ("Satya Nadella transformed Microsoft.", ["person", "organization"]),

    # Locations
    ("The conference is in New York City.", ["location", "event"]),
    ("Flight from London to Tokyo.", ["location"]),
    ("Meeting at the Paris office.", ["location"]),
    ("Vacation in the Bahamas.", ["location"]),
    ("Headquarters moved to San Francisco.", ["location", "organization"]),
    ("Born in Chicago, raised in Boston.", ["location"]),
    ("The restaurant is on 5th Avenue.", ["location"]),
    ("Traveling through Germany and France.", ["location"]),
    ("The beach in Miami is beautiful.", ["location"]),
    ("Office located in downtown Manhattan.", ["location"]),

    # Products and Items
    ("Buy iPhone and MacBook from Apple Store.", ["item", "organization", "location"]),
    ("The Tesla Model 3 is on sale.", ["item", "organization"]),
    ("Order PlayStation 5 from Amazon.", ["item", "organization"]),
    ("Install Windows 11 on the laptop.", ["item"]),
    ("The new Nike Air Max shoes.", ["item", "organization"]),
    ("Samsung Galaxy S24 review.", ["item", "organization"]),
    ("Bought a Rolex watch.", ["item", "organization"]),
    ("The IKEA furniture arrived.", ["item", "organization"]),
    ("Using Adobe Photoshop for editing.", ["item", "organization"]),
    ("The Sony headphones are great.", ["item", "organization"]),

    # Mixed entities
    ("I met Sarah at Starbucks in New York.", ["person", "organization", "location"]),
    ("Tom and Mary had dinner at the Italian restaurant in London.", ["person", "restaurant", "location"]),
    ("Dr. Johnson prescribed medication at General Hospital.", ["person", "organization", "item"]),
    ("CEO Lisa Su announced AMD's new chip.", ["person", "organization", "item"]),
    ("Professor Zhang teaches at Stanford in California.", ["person", "organization", "location"]),
    ("Chef Gordon Ramsay opened a restaurant in Las Vegas.", ["person", "restaurant", "location"]),
    ("Athlete LeBron James signed with the Lakers.", ["person", "organization"]),
    ("Author Stephen King lives in Maine.", ["person", "location"]),
    ("Singer Taylor Swift performed in Nashville.", ["person", "location"]),
    ("Director Christopher Nolan filmed in Iceland.", ["person", "location"]),

    # Food and Restaurants
    ("Order sushi from Nobu restaurant.", ["food", "restaurant"]),
    ("The pasta at Olive Garden was delicious.", ["food", "restaurant"]),
    ("Buy fresh salmon at Whole Foods.", ["food", "organization"]),
    ("Try the burgers at Five Guys.", ["food", "restaurant"]),
    ("The pizza from Domino's arrived.", ["food", "restaurant"]),
    ("Coffee from Starbucks every morning.", ["food", "organization"]),
    ("Tacos at Chipotle for lunch.", ["food", "restaurant"]),
    ("Ice cream from Ben & Jerry's.", ["food", "organization"]),
    ("Chicken sandwich at Chick-fil-A.", ["food", "restaurant"]),
    ("Breakfast at IHOP.", ["food", "restaurant"]),

    # Events and Dates
    ("The Olympics will be in Los Angeles.", ["event", "location"]),
    ("Super Bowl LVIII in Las Vegas.", ["event", "location"]),
    ("Coachella festival in California.", ["event", "location"]),
    ("Comic-Con in San Diego.", ["event", "location"]),
    ("The World Cup final in Qatar.", ["event", "location"]),
    ("Grammy Awards in Los Angeles.", ["event", "location"]),
    ("SXSW conference in Austin.", ["event", "location"]),
    ("CES in Las Vegas.", ["event", "location"]),
    ("Fashion Week in Milan.", ["event", "location"]),
    ("Oktoberfest in Munich.", ["event", "location"]),

    # Sports
    ("Michael Jordan played for the Chicago Bulls.", ["person", "organization", "location"]),
    ("Lionel Messi joined Inter Miami.", ["person", "organization"]),
    ("Serena Williams won at Wimbledon.", ["person", "event", "location"]),
    ("Tom Brady retired from the NFL.", ["person", "organization"]),
    ("Cristiano Ronaldo signed with Al-Nassr.", ["person", "organization"]),
    ("Tiger Woods won the Masters.", ["person", "event"]),
    ("Usain Bolt ran in the Olympics.", ["person", "event"]),
    ("Roger Federer retired from tennis.", ["person", "sport"]),
    ("Shohei Ohtani plays for the Dodgers.", ["person", "organization"]),
    ("Patrick Mahomes leads the Chiefs.", ["person", "organization"]),

    # Technology
    ("OpenAI released ChatGPT-5.", ["organization", "item"]),
    ("Google launched Gemini AI.", ["organization", "item"]),
    ("Apple announced the Vision Pro.", ["organization", "item"]),
    ("Microsoft acquired Activision.", ["organization"]),
    ("Netflix added new features.", ["organization"]),
    ("Spotify reached 500 million users.", ["organization"]),
    ("Twitter became X.", ["organization"]),
    ("Meta launched Threads.", ["organization", "item"]),
    ("NVIDIA stock hit record high.", ["organization"]),
    ("Amazon Web Services expanded.", ["organization"]),

    # Complex sentences
    (
        "Yesterday, John Smith from IBM met with Sarah Johnson at Google HQ in Mountain View.",
        ["person", "organization", "location"],
    ),
    (
        "The CEO of Tesla, Elon Musk, announced a new factory in Berlin, Germany.",
        ["person", "organization", "location"],
    ),
    (
        "Apple's Tim Cook and Microsoft's Satya Nadella attended the conference in Davos.",
        ["person", "organization", "location"],
    ),
    ("Dr. Anthony Fauci spoke about COVID-19 at the WHO meeting in Geneva.", ["person", "organization", "location"]),
    ("Amazon founder Jeff Bezos purchased the Washington Post newspaper.", ["person", "organization"]),
    ("Facebook, now Meta, moved its headquarters from Menlo Park.", ["organization", "location"]),
    ("The New York Times reported on events in Washington D.C.", ["organization", "location"]),
    ("SpaceX launched a rocket from Cape Canaveral, Florida.", ["organization", "location"]),
    ("Toyota and Honda announced electric vehicle plans in Tokyo.", ["organization", "location", "item"]),
    ("Netflix, Disney+, and HBO Max are competing in Los Angeles.", ["organization", "location"]),

    # Edge cases - short text
    ("Call John.", ["person"]),
    ("Buy milk.", ["item"]),
    ("Go to Paris.", ["location"]),
    ("Contact Microsoft.", ["organization"]),
    ("Meet at Starbucks.", ["organization", "location"]),
    ("Email Sarah.", ["person"]),
    ("Visit London.", ["location"]),
    ("Use Google.", ["organization"]),
    ("Try Netflix.", ["organization"]),
    ("Order from Amazon.", ["organization"]),
]
# fmt: on


def run_classification(
    model: "GLiNER2",
    text: str,
    labels: list[str],
    threshold: float = 0.0,
) -> dict[str, object]:
    """Run classification and return result with confidence."""
    result = model.classify_text(text, {"category": labels}, threshold=threshold, include_confidence=True)
    category_result = result["category"]
    return {
        "text": text,
        "labels": labels,
        "expected_label": category_result["label"],
        "expected_score": category_result["confidence"],
    }


def run_ner(
    model: "GLiNER2",
    text: str,
    labels: list[str],
    threshold: float = 0.5,
) -> dict[str, object]:
    """Run NER and return result with confidence and spans."""
    result = model.extract_entities(text, labels, threshold=threshold, include_confidence=True, include_spans=True)

    expected: list[dict[str, object]] = [
        {
            "text": entity["text"],
            "label": label,
            "score": entity["confidence"],
            "start": entity["start"],
            "end": entity["end"],
        }
        for label, entities in result.get("entities", {}).items()
        for entity in entities
    ]

    expected.sort(key=lambda x: (x["start"], x["end"], x["label"]))

    return {
        "text": text,
        "labels": labels,
        "threshold": threshold,
        "expected": expected,
    }


GLINER2_MODELS = [
    "fastino/gliner2-multi-v1",
    "fastino/gliner2-base-v1",
    "fastino/gliner2-large-v1",
]

GLINER1_MODELS = [
    "urchade/gliner_small-v2.1",
    "urchade/gliner_multi-v2.1",
    "urchade/gliner_large-v2.1",
]


def run_gliner1_ner(
    model: "GLiNER",
    text: str,
    labels: list[str],
    threshold: float = 0.5,
) -> dict[str, object]:
    """Run NER with GLiNER1 and return result with confidence and spans."""
    entities = model.predict_entities(text, labels, threshold=threshold)

    expected: list[dict[str, object]] = [
        {
            "text": entity["text"],
            "label": entity["label"],
            "score": entity["score"],
            "start": entity["start"],
            "end": entity["end"],
        }
        for entity in entities
    ]

    expected.sort(key=lambda x: (x["start"], x["end"], x["label"]))

    return {
        "text": text,
        "labels": labels,
        "threshold": threshold,
        "expected": expected,
    }


def run_gliner1_model_fixtures(model_name: str) -> dict[str, list[dict[str, object]]]:
    """Generate fixtures for a GLiNER1 model (NER only, no classification)."""
    from gliner import GLiNER

    print(f"\nLoading {model_name}...")
    model = GLiNER.from_pretrained(model_name)
    model.to("cuda")

    print(f"  Running {len(NER_TESTS)} NER tests...")
    ner_results = []
    for i, (text, labels) in enumerate(NER_TESTS):
        result = run_gliner1_ner(model, text, labels)
        ner_results.append(result)
        if (i + 1) % 50 == 0:
            print(f"    {i + 1}/{len(NER_TESTS)}")

    return {
        "ner": ner_results,
    }


def run_gliner2_model_fixtures(model_name: str) -> dict[str, list[dict[str, object]]]:
    """Generate fixtures for a GLiNER2 model (classification and NER)."""
    from gliner2 import GLiNER2

    print(f"\nLoading {model_name}...")
    model = GLiNER2.from_pretrained(model_name, device="cuda")
    model.eval()

    print(f"  Running {len(CLASSIFICATION_TESTS)} classification tests...")
    classification_results = []
    for i, (text, labels) in enumerate(CLASSIFICATION_TESTS):
        result = run_classification(model, text, labels)
        classification_results.append(result)
        if (i + 1) % 50 == 0:
            print(f"    {i + 1}/{len(CLASSIFICATION_TESTS)}")

    print(f"  Running {len(NER_TESTS)} NER tests...")
    ner_results = []
    for i, (text, labels) in enumerate(NER_TESTS):
        result = run_ner(model, text, labels)
        ner_results.append(result)
        if (i + 1) % 50 == 0:
            print(f"    {i + 1}/{len(NER_TESTS)}")

    return {
        "classification": classification_results,
        "ner": ner_results,
    }


def generate_gliner1_fixtures() -> None:
    """Generate fixtures for GLiNER1 ONNX models."""
    output_path = Path("tests/gliner1.fixtures.json")

    if output_path.exists():
        output_path.unlink()

    fixtures = {}
    for model_name in GLINER1_MODELS:
        model_key = model_name.split("/")[-1]
        fixtures[model_key] = run_gliner1_model_fixtures(model_name)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w") as f:
        json.dump(fixtures, f, indent=2)

    print(f"\n✓ Saved GLiNER1 fixtures for {len(GLINER1_MODELS)} models to {output_path}")


def generate_gliner2_fixtures() -> None:
    """Generate fixtures for GLiNER2 models."""
    output_path = Path("tests/gliner2.fixtures.json")

    if output_path.exists():
        output_path.unlink()

    fixtures = {}
    for model_name in GLINER2_MODELS:
        model_key = model_name.split("/")[-1]
        fixtures[model_key] = run_gliner2_model_fixtures(model_name)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w") as f:
        json.dump(fixtures, f, indent=2)

    print(f"\n✓ Saved GLiNER2 fixtures for {len(GLINER2_MODELS)} models to {output_path}")


def main() -> None:
    print("=" * 60)
    print("Generating GLiNER1 fixtures (NER only)")
    print("=" * 60)
    generate_gliner1_fixtures()

    print("\n" + "=" * 60)
    print("Generating GLiNER2 fixtures (Classification + NER)")
    print("=" * 60)
    generate_gliner2_fixtures()


if __name__ == "__main__":
    main()
