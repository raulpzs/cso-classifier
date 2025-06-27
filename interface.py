import gradio as gr
from classifier import classify_provision, classify_provision_isolated

def run_classifier(provision_text):
    result = classify_provision_isolated(provision_text)
    return result

gr.Interface(
    fn=run_classifier,
    inputs=gr.Textbox(lines=10, placeholder="Paste legal provision here:"),
    outputs="json",
    title="CSO Provision Classifier",
    description="Paste a legal provision to classify it using the CSO Matrix. Result includes ADICO interpretation, match, subgroup, and restrictiveness."
).launch()