python -m src.console storage download gs://amazon-revealed/Point-Cloud/01_ENTREGA_23_08_2023/NP/ACRE_005_NP_8973-536.laz

# Ler conteúdo direto da storage e imprimir no terminal
python -m src.console storage read gs://bucket/caminho/arquivo.txt

# Ler da storage e salvar localmente
python -m src.console storage read gs://bucket/caminho/arquivo.laz --output data/arquivo.laz

# Rodar app Streamlit (agora aceita Parquet direto de gs:// ou s3:// na sidebar)
streamlit run src/main.py
