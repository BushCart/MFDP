import gradio as gr
from scripts.query_engine_llm import generate_answer

def respond(user_message, chat_history):
    answer, confidence, sources = generate_answer(user_message)
    bot_message = (
        f"{answer}\n\n"
        f"{confidence}\n\n"
        f"Источники:\n" + "\n".join(sources)
    )
    return "", chat_history + [(user_message, bot_message)]

with gr.Blocks() as demo:
    gr.Markdown("## Askwise Chat")
    chatbot = gr.Chatbot()
    user_input = gr.Textbox(placeholder="Введите вопрос…", show_label=False)
    user_input.submit(respond, [user_input, chatbot], [user_input, chatbot])

if __name__ == "__main__":
    demo.launch()
