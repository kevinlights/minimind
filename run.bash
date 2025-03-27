# load=0: load from pytorch model, load=1: load from transformers-hf model
python eval_model.py --load 1 --model_mode 2 --device cpu
# python eval_model.py --load 1 --model_mode 2 --device mps

# pip install streamlit
# streamlit run web_demo.py
