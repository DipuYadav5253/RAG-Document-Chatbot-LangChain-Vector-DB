from app.rag_pipeline import ask_question_with_tracking

questions = [
    'What are Dipu skills?',
    'What is Dipu work experience?',
    'Which projects has Dipu built?'
]

for q in questions:
    print(f"\nAsking: {q}")
    result = ask_question_with_tracking(q)
    print(f"Answer: {result['answer']}")
    print(f"Response time: {result['response_time']:.2f} seconds")
    print("---")