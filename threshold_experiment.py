from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("all-MiniLM-L6-v2")

sentences = [
    # Group 1: Python
    "Python uses indentation to define the structure of code blocks.",
    "A Python list stores an ordered collection of items that can be changed after it is created.",
    "Functions in Python group reusable instructions behind a name and can accept arguments.",
    "A dictionary in Python stores values using unique keys for quick lookup.",
    "Python exceptions allow a program to handle errors without stopping unexpectedly.",
    # Group 2: Disc Golf
    "Disc golf players throw a disc from a tee area toward a metal basket.",
    "A stable disc golf drive usually depends on controlled footwork, balance, and a smooth release.",
    "Overstable discs tend to finish their flight to the left for a right-handed backhand throw.",
    "Disc golfers choose different discs for drives, approach shots, and short putts.",
    "Trees, hills, water, and wind can all change the strategy for a disc golf hole.",
    # Group 3: Dogs
    "Dogs use body language such as tail position, ear movement, and posture to communicate.",
    "Regular exercise helps many dogs stay healthy and reduces boredom-related behavior.",
    "Positive reinforcement rewards a dog for performing a desired behavior.",
    "Fresh water should be available to dogs throughout the day.",
    "A veterinarian can recommend preventive care based on a dog's age, breed, and health.",
    # Group 4: Cooking Thai Food
    "Thai cooking often balances sweet, sour, salty, spicy, and savory flavors.",
    "Lemongrass, galangal, kaffir lime leaves, and Thai basil add distinctive aromas to Thai dishes.",
    "Fish sauce provides a salty umami flavor in many Thai recipes.",
    "Coconut milk creates a rich base for curries such as green curry and red curry.",
    "Thai cooks often serve fresh herbs, lime, and chili as garnishes or flavor adjustments.",
]

embeddings = model.encode(sentences)

queries = [
    "What are some factors for choosing a disc in disc golf?",
    "How do I build an API in Python?",
    "What is the best way to start a curry base?",
    "What is the difference between running and jogging?",
    "What are some distinctive flavors for Thai cooking?",
    "What is Python?",
    "My dog's tail is wagging, is that a good thing?",
    "How many breeds of dog are there?",
]

thresholds = [0.3, 0.5, 0.7]

for query in queries:
    q_emb = model.encode(query)
    scores = util.cos_sim(q_emb, embeddings)[0]

    print(f"\nQuery: {query}")

    for threshold in thresholds:
        results = [
            (score.item(), sentence)
            for score, sentence in zip(scores, sentences)
            if score.item() >= threshold
        ]
        results.sort(reverse=True)

        print(f"\nThreshold {threshold}: {len(results)} results")
        for score, sentence in results:
            print(f"[{score:.2f}] {sentence}")
