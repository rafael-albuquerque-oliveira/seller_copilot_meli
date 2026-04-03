import polars as pl
import os


def process_reviews(
    input_csv_path: str, output_parquet_path: str, sample_size: int = 1000
) -> None:
    """
    Lê, limpa e extrai uma amostra das avaliações do Olist usando Polars.
    """
    print(f"Iniciando a leitura do arquivo: {input_csv_path}")

    # 1. Leitura Otimizada (Lazy Evaluation)
    # O scan_csv é mais eficiente que read_csv pois não carrega tudo na memória de uma vez
    query = pl.scan_csv(input_csv_path)

    # 2. Filtragem e Limpeza
    # Vamos manter apenas reviews que tenham texto na mensagem e uma nota
    query_cleaned = query.filter(pl.col("review_comment_message").is_not_null()).select(
        ["order_id", "review_score", "review_comment_message", "review_creation_date"]
    )

    # 3. Executa a query (collect)
    df = query_cleaned.collect()
    print(f"Total de reviews com texto encontrados: {df.height}")

    # 4. Amostragem
    # Extraímos uma amostra para não gastar muitos tokens/tempo na API de LLM depois
    df_sample = df.sample(n=sample_size, seed=42)

    # 5. Transformação de Strings
    # Padronizamos o texto para facilitar o RAG e o tagueamento
    df_final = df_sample.with_columns(
        pl.col("review_comment_message")
        .str.strip_chars()  # Remove espaços em branco nas pontas
        .str.replace_all(
            r"\n", " "
        )  # Remove quebras de linha que atrapalham a vetorização
        .alias("cleaned_message")
    )

    # 6. Salvando em Parquet
    # Parquet é um formato colunar, muito mais leve e rápido de ler que CSV
    os.makedirs(os.path.dirname(output_parquet_path), exist_ok=True)
    df_final.write_parquet(output_parquet_path)
    print(
        f"Amostra de {sample_size} linhas processada e salva em: {output_parquet_path}"
    )


if __name__ == "__main__":
    # Definindo os caminhos com base na estrutura do repositório
    INPUT_PATH = "../data/raw/olist_order_reviews_dataset.csv"
    OUTPUT_PATH = "../data/processed/cleaned_reviews.parquet"

    process_reviews(INPUT_PATH, OUTPUT_PATH)
