"""
Criação do banco de dados do Totem Inteligente FlexMedia.

Estrutura relacional com 3 tabelas:
- sessoes: cada visita de um usuário ao totem
- interacoes: cada ação/escolha dentro de uma sessão
- recomendacoes: sugestão feita pelo totem e resposta do usuário

Inclui chaves estrangeiras, constraints de validação e índices.
"""

import sqlite3
import os


def criar_banco(caminho_db: str = None) -> None:
    """Cria o banco de dados com todas as tabelas e constraints."""

    if caminho_db is None:
        # resolve o caminho relativo à raiz do projeto
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        caminho_db = os.path.join(base_dir, "data", "interacoes.db")

    # garante que a pasta data/ existe
    os.makedirs(os.path.dirname(caminho_db), exist_ok=True)

    # remove banco antigo para recriar do zero
    if os.path.exists(caminho_db):
        os.remove(caminho_db)
        print(f"Banco anterior removido: {caminho_db}")

    conn = sqlite3.connect(caminho_db)

    # habilita suporte a chaves estrangeiras (desabilitado por padrão no SQLite)
    conn.execute("PRAGMA foreign_keys = ON;")

    cursor = conn.cursor()

    # =========================================================================
    # TABELA: sessoes
    # Representa cada visita de um usuário ao totem.
    # =========================================================================
    cursor.execute("""
        CREATE TABLE sessoes (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            inicio_sessao   TEXT    NOT NULL,
            fim_sessao      TEXT    NOT NULL,
            faixa_etaria    TEXT    NOT NULL CHECK (faixa_etaria IN (
                                        'jovem', 'adulto', 'idoso'
                                   )),
            dia_semana      TEXT    NOT NULL CHECK (dia_semana IN (
                                        'segunda', 'terca', 'quarta',
                                        'quinta', 'sexta', 'sabado', 'domingo'
                                   )),
            faixa_horaria   TEXT    NOT NULL CHECK (faixa_horaria IN (
                                        'manha', 'almoco', 'tarde', 'noite'
                                   ))
        );
    """)

    # =========================================================================
    # TABELA: interacoes
    # Cada ação/escolha que o usuário faz dentro de uma sessão.
    # Uma sessão pode ter múltiplas interações.
    # =========================================================================
    cursor.execute("""
        CREATE TABLE interacoes (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            sessao_id       INTEGER NOT NULL,
            timestamp       TEXT    NOT NULL,
            categoria       TEXT    NOT NULL CHECK (categoria IN (
                                        'comer', 'comprar', 'descansar', 'lazer'
                                   )),
            preferencia     TEXT    NOT NULL,
            tempo_interacao INTEGER NOT NULL CHECK (tempo_interacao > 0),

            FOREIGN KEY (sessao_id) REFERENCES sessoes(id)
                ON DELETE CASCADE
                ON UPDATE CASCADE
        );
    """)

    # =========================================================================
    # TABELA: recomendacoes
    # O que o totem sugeriu para cada interação e se o usuário aceitou.
    # =========================================================================
    cursor.execute("""
        CREATE TABLE recomendacoes (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            interacao_id        INTEGER NOT NULL,
            loja_recomendada    TEXT    NOT NULL,
            aceitou             INTEGER NOT NULL CHECK (aceitou IN (0, 1)),
            motivo_rejeicao     TEXT    DEFAULT NULL,

            FOREIGN KEY (interacao_id) REFERENCES interacoes(id)
                ON DELETE CASCADE
                ON UPDATE CASCADE
        );
    """)

    # =========================================================================
    # ÍNDICES para consultas frequentes
    # =========================================================================
    cursor.execute("""
        CREATE INDEX idx_interacoes_sessao
        ON interacoes(sessao_id);
    """)

    cursor.execute("""
        CREATE INDEX idx_interacoes_categoria
        ON interacoes(categoria);
    """)

    cursor.execute("""
        CREATE INDEX idx_recomendacoes_interacao
        ON recomendacoes(interacao_id);
    """)

    cursor.execute("""
        CREATE INDEX idx_sessoes_faixa_horaria
        ON sessoes(faixa_horaria);
    """)

    cursor.execute("""
        CREATE INDEX idx_sessoes_dia_semana
        ON sessoes(dia_semana);
    """)

    conn.commit()
    conn.close()

    print("=" * 50)
    print("BANCO DE DADOS CRIADO COM SUCESSO")
    print("=" * 50)
    print(f"\nCaminho: {caminho_db}")
    print("\nTabelas criadas:")
    print("  - sessoes        (visitas ao totem)")
    print("  - interacoes     (ações do usuário)")
    print("  - recomendacoes  (sugestões e respostas)")
    print("\nConstraints aplicadas:")
    print("  - Foreign Keys com ON DELETE CASCADE")
    print("  - CHECK em faixa_etaria, dia_semana, faixa_horaria")
    print("  - CHECK em categoria, tempo_interacao > 0, aceitou IN (0,1)")
    print("\nÍndices criados:")
    print("  - idx_interacoes_sessao")
    print("  - idx_interacoes_categoria")
    print("  - idx_recomendacoes_interacao")
    print("  - idx_sessoes_faixa_horaria")
    print("  - idx_sessoes_dia_semana")


if __name__ == "__main__":
    criar_banco()