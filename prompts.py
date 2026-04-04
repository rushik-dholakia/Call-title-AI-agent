# ===============================
# Ask Clarifying Questions
# ===============================
# ===============================
# prompts.py
# ===============================

def generate_title_prompt(transcript, similar_titles):
    return f"""
You are an expert AI assistant for customer support.

Your task is to generate a short, clear, and professional call title.

Call Transcript:
{transcript}

Similar Titles:
{similar_titles}

Rules:
- Maximum 6 words
- Focus on the main issue
- Use professional/business language
- Avoid unnecessary words
- Do NOT repeat similar titles exactly

Return ONLY the final call title.
"""


def extract_facts_prompt(text):
    return f"""
Extract key facts from the user issue.

Text:
{text}

Return:
- Issue Type
- Device/System
- Error (if any)
- Keywords
- Root Cause

Be concise.
"""

def confidence_prompt(context):
    return f"""
You are evaluating whether enough information is available
to identify the correct call title.

Context:
{context}

IMPORTANT:
- Return ONLY a number between 0 and 1
- Do NOT explain anything
- Do NOT write text
- Example output: 0.82

Your output:
"""




def ask_questions_prompt(transcript):
    return f"""
You are an intelligent AI agent performing root cause analysis.

Based on the following call transcript, ask ONLY ONE most important
clarifying question to better understand the issue.

Transcript:
{transcript}

Rules:
- Ask only ONE question
- The question must be highly relevant
- Keep it short and clear
- Focus on identifying the root cause

Return ONLY the question.
"""

def select_best_title_prompt(context, similar_titles):
    return f"""
You are an expert AI assistant.

Context:
{context}

Available Titles:
{similar_titles}

VERY IMPORTANT:
- If the issue is related to network or connectivity, prefer VPN-related titles
- Do NOT select generic "App Crash" unless clearly mentioned
- Choose the most specific match

Return ONLY one title from the list.
"""

def ask_next_question_prompt(context):
    return f"""
You are an AI agent performing root cause analysis.

Based on the conversation so far, ask ONE next best question
to better understand the issue.

Conversation:
{context}

Rules:
- Ask only ONE question
- If enough information is available, return: DONE

Return either:
- A question
OR
- DONE
"""