import gradio as gr
from classifier import classify_provision, classify_provision_isolated
from matrix_first_codings import code_law

def run_classifier(provision_text):
    result = classify_provision_isolated(provision_text)
    return result

def run_coder(law_text):
    result = code_law(law_text)
    return result

gr.Interface(
    #fn=run_classifier,
    fn=run_coder,
    inputs=gr.Textbox(lines=10, placeholder="Paste here:"),
    outputs="json",
    #title="CSO Provision Classifier",
    #description="Paste a legal provision to classify it using the CSO Matrix. Result includes ADICO interpretation, match, subgroup, and restrictiveness."
    title="CSO Law Coder",
    description="Paste a legal text to code it using the CSO Matrix. Result includes law metadata and provision codings."
).launch()