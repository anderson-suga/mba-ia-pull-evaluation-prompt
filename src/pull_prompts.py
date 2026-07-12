"""
Script para fazer pull de prompts do LangSmith Prompt Hub.

Este script:
1. Conecta ao LangSmith usando credenciais do .env
2. Faz pull dos prompts do Hub
3. Salva localmente em prompts/bug_to_user_story_v1.yml

SIMPLIFICADO: Usa serialização nativa do LangChain para extrair prompts.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from langchain import hub
from langsmith import Client
from utils import save_yaml, check_env_vars, print_section_header

load_dotenv()

HUB_PROMPT_ID = "leonanluppi/bug_to_user_story_v1"
LOCAL_PROMPT_NAME = "bug_to_user_story_v1"
OUTPUT_FILE = Path(__file__).parent.parent / "prompts" / "bug_to_user_story_v1.yml"

# Valores usados quando o Hub não fornece metadados para o prompt
DEFAULT_METADATA = {
    "description": "Prompt para converter relatos de bugs em User Stories",
    "version": "v1",
    "created_at": "",
    "tags": ["bug-analysis", "user-story", "product-management"],
}


def extract_message_text(message) -> str:
    """
    Extrai o texto de uma mensagem de um ChatPromptTemplate.

    O Hub pode retornar mensagens em formatos diferentes:
    - Templates de mensagem (ex: SystemMessagePromptTemplate) expõem o texto
      em `.prompt.template`
    - Mensagens simples (ex: SystemMessage) expõem o texto em `.content`

    Args:
        message: Objeto de mensagem do LangChain

    Returns:
        Texto da mensagem ou string vazia se não for possível extrair
    """
    prompt_attr = getattr(message, "prompt", None)
    if prompt_attr is not None and hasattr(prompt_attr, "template"):
        return prompt_attr.template

    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content

    return ""


def split_system_and_user_prompts(template) -> tuple:
    """
    Separa o system_prompt e o user_prompt de um template puxado do Hub.

    Percorre as mensagens do template identificando o papel de cada uma
    pelo nome da classe (ex: "SystemMessagePromptTemplate",
    "HumanMessagePromptTemplate") ou pelo atributo `role`, sem assumir
    uma ordem ou quantidade fixa de mensagens.

    Args:
        template: ChatPromptTemplate (ou PromptTemplate) retornado pelo Hub

    Returns:
        (system_prompt, user_prompt) - Textos extraídos (vazios se ausentes)
    """
    system_prompt = ""
    user_prompt = ""

    # PromptTemplate simples não tem `.messages`; nesse caso trata o próprio
    # template como uma única mensagem de usuário
    messages = getattr(template, "messages", None)
    if messages is None:
        return "", getattr(template, "template", "")

    for message in messages:
        class_name = type(message).__name__
        role = getattr(message, "role", "") or ""
        text = extract_message_text(message)

        if "System" in class_name or role == "system":
            system_prompt = text
        elif "Human" in class_name or "User" in class_name or role in ("human", "user"):
            user_prompt = text

    return system_prompt, user_prompt


def fetch_prompt_metadata(client: Client, prompt_id: str) -> dict:
    """
    Busca metadados do prompt (descrição, tags, data de criação) na API do
    LangSmith. Se algum campo estiver indisponível, usa valores padrão.

    Args:
        client: Cliente LangSmith autenticado
        prompt_id: Identificador do prompt no Hub (ex: "owner/nome")

    Returns:
        Dicionário com description, version, created_at e tags
    """
    metadata = dict(DEFAULT_METADATA)

    try:
        prompt_info = client.get_prompt(prompt_id)

        if prompt_info is not None:
            if getattr(prompt_info, "description", None):
                metadata["description"] = prompt_info.description
            if getattr(prompt_info, "tags", None):
                metadata["tags"] = list(prompt_info.tags)
            created_at = getattr(prompt_info, "created_at", None)
            if created_at:
                # datetime não é serializável de forma legível no YAML;
                # guarda apenas a data (YYYY-MM-DD)
                metadata["created_at"] = str(created_at)[:10]

    except Exception as e:
        print(f"   ⚠️  Não foi possível obter metadados do Hub ({e}). Usando valores padrão.")

    return metadata


def pull_prompts_from_langsmith() -> bool:
    """
    Faz pull do prompt do LangSmith Hub e salva em YAML local.

    Returns:
        True se sucesso, False caso contrário
    """
    print(f"Puxando prompt do LangSmith Hub: {HUB_PROMPT_ID}...")

    try:
        template = hub.pull(HUB_PROMPT_ID)
    except Exception as e:
        print(f"❌ Erro ao puxar prompt '{HUB_PROMPT_ID}': {e}")
        print("\nVerifique:")
        print("- LANGSMITH_API_KEY está configurada corretamente no .env")
        print("- Sua conexão com a internet está funcionando")
        return False

    print("   ✓ Prompt carregado com sucesso")

    system_prompt, user_prompt = split_system_and_user_prompts(template)

    if not system_prompt and not user_prompt:
        print("❌ Não foi possível extrair system_prompt/user_prompt do template retornado.")
        return False

    client = Client()
    metadata = fetch_prompt_metadata(client, HUB_PROMPT_ID)

    prompt_data = {
        LOCAL_PROMPT_NAME: {
            "description": metadata["description"],
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "version": metadata["version"],
            "created_at": metadata["created_at"],
            "tags": metadata["tags"],
        }
    }

    if not save_yaml(prompt_data, str(OUTPUT_FILE)):
        return False

    print(f"   ✓ Prompt salvo em: {OUTPUT_FILE}")
    return True


def main():
    """Função principal"""
    print_section_header("PULL DE PROMPTS DO LANGSMITH HUB")

    if not check_env_vars(["LANGSMITH_API_KEY"]):
        return 1

    if not pull_prompts_from_langsmith():
        return 1

    print("\n✅ Pull concluído com sucesso!")
    print("\nPróximos passos:")
    print("1. Analise o prompt em prompts/bug_to_user_story_v1.yml")
    print("2. Crie sua versão otimizada em prompts/bug_to_user_story_v2.yml")
    print("3. Faça push da versão otimizada: python src/push_prompts.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
