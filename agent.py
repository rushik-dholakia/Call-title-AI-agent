import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import ollama
import re


from prompts import (
    ask_next_question_prompt,
    select_best_title_prompt,
    extract_facts_prompt,
    confidence_prompt
)

# ===============================
# Load models
# ===============================
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
index = faiss.read_index("faiss_index.index")

with open("titles.pkl", "rb") as f:
    titles = pickle.load(f)

print("✅ Smart RCA Agent Ready!\n")

# ===============================
# FAISS Search
# ===============================
def search_similar_titles(query, k=5):
    query_embedding = embedding_model.encode([query]).astype("float32")
    distances, indices = index.search(query_embedding, k)

    results = []
    seen = set()

    for i in indices[0]:
        if i < len(titles):
            title = titles[i]
            if title not in seen:
                results.append(title)
                seen.add(title)

    return results


# ===============================
# Keyword Boost
# ===============================
def boost_titles(query, titles_list):
    query = query.lower()

    priority_keywords = {
        "vpn": ["vpn", "network", "internet", "connect"],
        "pc": ["pc", "computer", "system"],
        "payment": ["payment", "refund", "transaction"],
        "app crash": ["crash", "error", "bug", "issue", "app not opening"],
        "password reset": ["password", "reset", "forgot"],
        "New Domain ID Creation": ["domain", "id", "creation", "new"],
        "New Gate Pass request": ["gate", "pass", "request", "new"],
        "ID Card not accessing": ["id card", "accessing", "unregistered", "employee", "visitor"]
    }

    boosted = []
    others = []

    for title in titles_list:
        title_lower = title.lower()

        if any(word in query for word in priority_keywords.get("vpn", [])) and "vpn" in title_lower:
            boosted.append(title)
        else:
            others.append(title)

    return boosted + others


# ===============================
# Extract Facts
# ===============================
def extract_facts(text):
    prompt = extract_facts_prompt(text)

    response = ollama.chat(
        model="llama3",
        messages=[{"role": "user", "content": prompt}]
    )

    return response['message']['content']


# ===============================
# Ask Next Question
# ===============================
def ask_next_question(context):
    prompt = ask_next_question_prompt(context)

    response = ollama.chat(
        model="llama3",
        messages=[{"role": "user", "content": prompt}]
    )

    return response['message']['content'].strip()


# ===============================
# Confidence Calculation
# ===============================
def get_confidence(context):
    prompt = confidence_prompt(context)

    response = ollama.chat(
        model="llama3",
        messages=[{"role": "user", "content": prompt}]
    )

    text = response['message']['content'].strip()

    # 🔥 Extract number using regex
    match = re.search(r"\d*\.?\d+", text)

    if match:
        value = float(match.group())

        # Clamp between 0 and 1
        return max(0.0, min(1.0, value))

    return 0.5


# ===============================
# Select Best Title
# ===============================
def select_best_title(context, similar_titles):
    prompt = select_best_title_prompt(context, similar_titles)

    response = ollama.chat(
        model="llama3",
        messages=[{"role": "user", "content": prompt}]
    )

    result = response['message']['content'].strip()

    if result not in similar_titles:
        return similar_titles[0]

    return result


# ===============================
# MAIN AGENT LOOP
# ===============================
def run_agent():
    while True:
        user_input = input("\n📞 Describe the issue (or type 'exit'): ")

        if user_input.lower() == "exit":
            print(" Exiting agent...")
            break

        # 🔹 Step 1: Extract facts
        facts = extract_facts(user_input)
        print("\n🧠 Extracted Facts:")
        print(facts)

        context = f"User: {user_input}\nFacts: {facts}\n"

        confidence = 0
        threshold = 0.80
        max_questions = 10
        question_count = 0

        # 🔥 Dynamic questioning loop
        while confidence < threshold and question_count < max_questions:

            next_q = ask_next_question(context)

            if next_q.upper() == "DONE":
                print("\n🤖 Agent: Enough information gathered.")
                break

            print("\n🤖 Question:", next_q)

            answer = input("✍️ Your answer: ")

            context += f"Q: {next_q}\nA: {answer}\n"

            # 🔥 Update confidence
            confidence = get_confidence(context)
            print(f"📊 Confidence: {confidence:.2f}")

            question_count += 1

        # 🔹 Step 2: FAISS search
        similar_titles = search_similar_titles(context, k=10)
        similar_titles = boost_titles(context, similar_titles)

        print("\n🔍 Similar Titles:")
        for t in similar_titles:
            print("-", t)

        # 🔹 Step 3: Final title selection
        final_title = select_best_title(context, similar_titles)

        print("\n🎯 Final Call Title:")
        print(final_title)

        #print("\n" + "="*50)


# ===============================
# Run
# ===============================
if __name__ == "__main__":
    run_agent()