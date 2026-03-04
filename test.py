from app.rag_pipeline import ask_question

questions = [
    'What is Dipu work experience?',
    'Which projects has Dipu built?'
]

for q in questions:
    result = ask_question(q)
    print(f"Q: {q}")
    print(f"A: {result['answer']}")
    print("---")