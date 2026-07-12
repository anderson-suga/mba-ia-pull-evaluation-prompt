"""
Script para fazer push de prompts otimizados ao LangSmith Prompt Hub.

Este script:
1. Lê os prompts otimizados de prompts/bug_to_user_story_v2.yml
2. Valida os prompts
3. Faz push PÚBLICO para o LangSmith Hub
4. Adiciona metadados (tags, descrição, técnicas utilizadas)

SIMPLIFICADO: Código mais limpo e direto ao ponto.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from langsmith import Client
from langchain_core.prompts import ChatPromptTemplate
from utils import load_yaml, check_env_vars, print_section_header, validate_prompt_structure

load_dotenv()

PROMPT_NAME = "bug_to_user_story_v2"
PROMPT_FILE = Path(__file__).parent.parent / "prompts" / "bug_to_user_story_v2.yml"


def technique_to_tag(technique: str) -> str:
    """
    Converte o nome de uma técnica em tag no formato do Hub.
    Ex: "Few-shot Learning" -> "few-shot-learning"
    """
    return technique.strip().lower().replace(" ", "-")


def build_hub_tags(prompt_data: dict) -> list:
    """
    Monta a lista de tags a publicar no Hub.

    A API do LangSmith não aceita metadados arbitrários no push, então as
    técnicas de prompt engineering aplicadas são convertidas em tags para
    ficarem visíveis na página pública do prompt.

    Args:
        prompt_data: Dados do prompt (dict interno do YAML)

    Returns:
        Lista de tags sem duplicatas, preservando a ordem
    """
    tags = list(prompt_data.get("tags", []))

    for technique in prompt_data.get("techniques_applied", []):
        tag = technique_to_tag(technique)
        if tag not in tags:
            tags.append(tag)

    return tags


def build_hub_description(prompt_data: dict) -> str:
    """
    Monta a descrição a publicar no Hub, incluindo as técnicas aplicadas
    em texto livre (também por falta de campo de metadados arbitrários).

    Args:
        prompt_data: Dados do prompt (dict interno do YAML)

    Returns:
        Descrição completa para a página pública do prompt
    """
    description = prompt_data.get("description", "")
    techniques = prompt_data.get("techniques_applied", [])

    if techniques:
        description += f" Técnicas aplicadas: {', '.join(techniques)}."

    return description


def push_prompt_to_langsmith(prompt_name: str, prompt_data: dict) -> bool:
    """
    Faz push do prompt otimizado para o LangSmith Hub (PÚBLICO).

    Args:
        prompt_name: Nome do prompt
        prompt_data: Dados do prompt

    Returns:
        True se sucesso, False caso contrário
    """
    username = os.getenv("USERNAME_LANGSMITH_HUB", "")
    full_prompt_name = f"{username}/{prompt_name}"

    template = ChatPromptTemplate.from_messages([
        ("system", prompt_data["system_prompt"]),
        ("human", prompt_data["user_prompt"]),
    ])

    print(f"Fazendo push do prompt: {full_prompt_name}...")

    try:
        client = Client()
        url = client.push_prompt(
            full_prompt_name,
            object=template,
            description=build_hub_description(prompt_data),
            tags=build_hub_tags(prompt_data),
            is_public=True,
        )
    except Exception as e:
        print(f"❌ Erro ao fazer push do prompt: {e}")

        # Publicar um prompt público exige que a conta já tenha um "handle"
        # no Hub, criado uma única vez pela interface web
        if "handle" in str(e).lower():
            print("\nSua conta LangSmith ainda não tem um handle no Hub.")
            print("Publique qualquer prompt manualmente uma vez em:")
            print("  https://smith.langchain.com/prompts")
            print("Depois configure USERNAME_LANGSMITH_HUB no .env e rode este script novamente.")

        return False

    print("   ✓ Push realizado com sucesso (público)")
    print(f"   ✓ URL: {url}")
    return True


def validate_prompt(prompt_data: dict) -> tuple[bool, list]:
    """
    Valida estrutura básica de um prompt (versão simplificada).

    Args:
        prompt_data: Dados do prompt

    Returns:
        (is_valid, errors) - Tupla com status e lista de erros
    """
    is_valid, errors = validate_prompt_structure(prompt_data)

    if not str(prompt_data.get("user_prompt", "")).strip():
        errors.append("user_prompt está vazio")
        is_valid = False

    if "{bug_report}" not in str(prompt_data.get("user_prompt", "")):
        errors.append("user_prompt não contém a variável {bug_report}")
        is_valid = False

    return (is_valid, errors)


def main():
    """Função principal"""
    print_section_header("PUSH DE PROMPTS OTIMIZADOS AO LANGSMITH HUB")

    if not check_env_vars(["LANGSMITH_API_KEY", "USERNAME_LANGSMITH_HUB"]):
        return 1

    data = load_yaml(str(PROMPT_FILE))
    if not data:
        return 1

    # O YAML pode ter os campos aninhados sob a chave do prompt
    # (mesmo layout da v1) ou direto na raiz; suporta os dois formatos
    prompt_data = data.get(PROMPT_NAME, data)

    is_valid, errors = validate_prompt(prompt_data)
    if not is_valid:
        print("❌ Prompt inválido:")
        for error in errors:
            print(f"   - {error}")
        return 1

    print("   ✓ Estrutura do prompt validada")

    if not push_prompt_to_langsmith(PROMPT_NAME, prompt_data):
        return 1

    print("\n✅ Push concluído com sucesso!")
    print("\nPróximos passos:")
    print("1. Confirme que o prompt está público em: https://smith.langchain.com/prompts")
    print("2. Execute a avaliação: python src/evaluate.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
