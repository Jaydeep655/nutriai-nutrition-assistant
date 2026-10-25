"""
NutriAI — Food Image Recognition + Nutrition Chatbot
------------------------------------------------------
Real working version of the project: upload a food photo, get it classified
by a pretrained vision transformer, look up its nutrition profile, and ask a
chatbot follow-up questions about your daily intake.

Run locally:
    pip install -r requirements.txt
    python app.py           # launches a local Gradio link
    # or deploy directly to a Hugging Face Space (drop these 2 files in a Space repo)

Uses:
- `nateraw/food` (ViT fine-tuned on Food-101) for image classification — no API key needed
- A small local nutrition lookup table (swap for a real API like USDA FoodData Central if you want)
- A lightweight rule-based chatbot; swap `answer_question()` for a call to any LLM API for a
  true GenAI chatbot (see the commented-out OpenAI/Anthropic snippet below)
"""

import gradio as gr
from transformers import pipeline

classifier = pipeline("image-classification", model="nateraw/food")

# Minimal nutrition table (per 100g, approximate) — extend this or swap for a real API
NUTRITION_DB = {
    "pizza": {"calories": 266, "protein_g": 11, "carbs_g": 33, "fat_g": 10},
    "sushi": {"calories": 150, "protein_g": 6, "carbs_g": 30, "fat_g": 0.7},
    "hamburger": {"calories": 295, "protein_g": 17, "carbs_g": 24, "fat_g": 14},
    "ice_cream": {"calories": 207, "protein_g": 3.5, "carbs_g": 24, "fat_g": 11},
    "salad": {"calories": 152, "protein_g": 5, "carbs_g": 10, "fat_g": 10},
    "steak": {"calories": 271, "protein_g": 25, "carbs_g": 0, "fat_g": 19},
    "french_fries": {"calories": 312, "protein_g": 3.4, "carbs_g": 41, "fat_g": 15},
    "default": {"calories": 200, "protein_g": 8, "carbs_g": 25, "fat_g": 8},
}

daily_log = []


def classify_and_lookup(image):
    if image is None:
        return "Upload a food photo first.", None
    preds = classifier(image)
    top = preds[0]
    label = top["label"].lower().replace(" ", "_")
    confidence = top["score"]

    info = NUTRITION_DB.get(label, NUTRITION_DB["default"])
    daily_log.append({"food": label, **info})

    summary = (
        f"**Detected:** {top['label']} ({confidence*100:.1f}% confidence)\n\n"
        f"**Estimated nutrition (per 100g):**\n"
        f"- Calories: {info['calories']} kcal\n"
        f"- Protein: {info['protein_g']} g\n"
        f"- Carbs: {info['carbs_g']} g\n"
        f"- Fat: {info['fat_g']} g"
    )
    return summary, build_daily_chart()


def build_daily_chart():
    if not daily_log:
        return None
    import pandas as pd
    df = pd.DataFrame(daily_log)
    totals = df[["calories", "protein_g", "carbs_g", "fat_g"]].sum()
    return totals.to_frame(name="today's total").reset_index().rename(columns={"index": "metric"})


def answer_question(message, history):
    """Simple rule-based nutrition assistant.

    For a true GenAI chatbot, replace this function body with an API call, e.g.:

        from anthropic import Anthropic
        client = Anthropic()
        resp = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=300,
            messages=[{"role": "user", "content": f"You are a nutrition coach. {message}"}],
        )
        return resp.content[0].text
    """
    msg = message.lower()
    if not daily_log:
        return "Log a food photo first so I have something to analyze!"

    import pandas as pd
    df = pd.DataFrame(daily_log)
    totals = df[["calories", "protein_g", "carbs_g", "fat_g"]].sum()

    if "calorie" in msg:
        return f"You've logged {totals['calories']:.0f} kcal today across {len(daily_log)} item(s)."
    if "protein" in msg:
        return f"You've had about {totals['protein_g']:.1f}g of protein today."
    if "recommend" in msg or "suggest" in msg:
        if totals["protein_g"] < 40:
            return "Your protein intake looks low today — consider adding eggs, legumes, or grilled chicken."
        return "Your macros look reasonably balanced so far. Keep it up!"
    return "I can tell you about your calories, protein, carbs or fat so far today — just ask."


with gr.Blocks(title="NutriAI") as demo:
    gr.Markdown("# 🥗 NutriAI — Food Recognition + Nutrition Chatbot")
    gr.Markdown("Upload a meal photo for instant nutrition analysis, then chat about your daily intake.")

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Upload food photo")
            classify_btn = gr.Button("Analyze", variant="primary")
            result_md = gr.Markdown()
        with gr.Column():
            chart = gr.Dataframe(label="Today's running totals")

    classify_btn.click(classify_and_lookup, inputs=image_input, outputs=[result_md, chart])

    gr.Markdown("### Ask the nutrition assistant")
    gr.ChatInterface(answer_question)

if __name__ == "__main__":
    demo.launch()
