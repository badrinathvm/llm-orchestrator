ROUTE_EXAMPLES: dict[str, list[str]] = {
    "small": [
        "Hi, how are you?",
        "What's the capital of France?",
        "What time is it in Tokyo right now?",
        "Define the word 'ubiquitous'.",
        "How many days are there in a leap year?",
        "Translate 'good morning' to Spanish.",
    ],
    "frontier": [
        "Can you analyze the trade-offs between microservices and monolithic architectures for a fast-growing startup?",
        "Walk me through how to design a distributed rate limiter that scales across regions.",
        "Compare and contrast three different approaches to solving the traveling salesman problem and explain their complexity trade-offs.",
        "Help me draft a technical strategy for migrating a legacy monolith to event-driven microservices over 18 months.",
        "Explain the implications of CAP theorem on a multi-region database design and recommend an approach for our use case.",
        "What's the best way to architect a system that needs strong consistency but also high availability across continents?",
    ],
    "specialist": [
        "My account got charged twice this month, can you help me get a refund?",
        "I can't log into my account, it keeps saying invalid password even after I reset it.",
        "The app keeps crashing every time I try to upload a photo, how do I fix this?",
        "I want to cancel my subscription but the cancel button isn't working.",
        "My order shipped to the wrong address, what should I do now?",
        "I'm getting an error code E-2048 when I try to check out, what does that mean?",
    ],
}
