"""
Testes automatizados para validação de prompts.
"""
import re
import sys
from pathlib import Path

import pytest
import yaml

# Adicionar src ao path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils import validate_prompt_structure

PROMPT_FILE = Path(__file__).parent.parent / "prompts" / "bug_to_user_story_v2.yml"
PROMPT_KEY = "bug_to_user_story_v2"

def load_prompts(file_path: str):
    """Carrega prompts do arquivo YAML."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

@pytest.fixture
def prompt_data():
    """
    Carrega o prompt v2 e retorna o dicionário de configuração.

    Aceita tanto o layout aninhado (campos sob a chave 'bug_to_user_story_v2',
    mesmo formato da v1) quanto o layout flat (campos na raiz do arquivo).
    Valida a estrutura básica antes de retornar, para que qualquer teste
    falhe imediatamente com uma mensagem clara se o YAML estiver quebrado.
    """
    data = load_prompts(str(PROMPT_FILE))
    assert data, f"Arquivo YAML vazio ou inválido: {PROMPT_FILE}"

    config = data.get(PROMPT_KEY, data)

    is_valid, errors = validate_prompt_structure(config)
    assert is_valid, f"Estrutura básica do prompt inválida: {errors}"

    return config

class TestPrompts:
    def test_prompt_has_system_prompt(self, prompt_data):
        """Verifica se o campo 'system_prompt' existe e não está vazio."""
        assert "system_prompt" in prompt_data, "Campo 'system_prompt' não encontrado"
        assert prompt_data["system_prompt"].strip(), "'system_prompt' está vazio"

    def test_prompt_has_role_definition(self, prompt_data):
        """Verifica se o prompt define uma persona (ex: "Você é um Product Manager")."""
        content = prompt_data["system_prompt"].lower()

        role_keywords = [
            "você é um",
            "você é uma",
            "you are a",
            "you are an",
            "product manager",
            "product owner",
            "atuando como",
            "acting as",
        ]

        assert any(keyword in content for keyword in role_keywords), (
            "Nenhuma definição de persona encontrada no system_prompt "
            f"(esperado algo como: {role_keywords})"
        )

    def test_prompt_mentions_format(self, prompt_data):
        """Verifica se o prompt exige formato Markdown ou User Story padrão."""
        content = prompt_data["system_prompt"].lower()

        format_keywords = [
            "como um",       # template "Como um..., eu quero..., para que..."
            "eu quero",
            "para que",
            "dado que",      # critérios de aceitação em Gherkin/BDD
            "quando",
            "então",
            "given",
            "when",
            "then",
            "user story",
            "markdown",
        ]

        assert any(keyword in content for keyword in format_keywords), (
            "Nenhuma menção a formato de User Story ou Markdown encontrada "
            f"(esperado algo como: {format_keywords})"
        )

    def test_prompt_has_few_shot_examples(self, prompt_data):
        """Verifica se o prompt contém exemplos de entrada/saída (técnica Few-shot)."""
        content = prompt_data["system_prompt"].lower()

        example_markers = ["exemplo", "example", "few-shot"]
        input_markers = ["relato de bug", "bug report", "input", "entrada"]
        output_markers = ["user story esperada", "expected output", "output", "saída"]

        assert any(marker in content for marker in example_markers), (
            "Nenhum marcador de exemplo (Few-shot) encontrado no system_prompt"
        )
        assert any(marker in content for marker in input_markers), (
            "Exemplos Few-shot não contêm entrada (relato de bug)"
        )
        assert any(marker in content for marker in output_markers), (
            "Exemplos Few-shot não contêm saída esperada (user story)"
        )

    def test_prompt_no_todos(self, prompt_data):
        """Garante que você não esqueceu nenhum `[TODO]` no texto."""
        # Verificação case-sensitive: em maiúsculas pega apenas marcadores
        # temporários, sem confundir com palavras comuns em português
        # como "todo"/"todos"
        temp_markers = ["TODO", "FIXME", "TBD", "XXX"]

        fields_to_check = ["system_prompt", "user_prompt", "description"]

        for field in fields_to_check:
            content = str(prompt_data.get(field, ""))
            for marker in temp_markers:
                assert marker not in content, (
                    f"Marcador temporário '{marker}' encontrado no campo '{field}'"
                )

    def test_minimum_techniques(self, prompt_data):
        """Verifica (através dos metadados do yaml) se pelo menos 2 técnicas foram listadas."""
        techniques = prompt_data.get("techniques_applied")

        assert isinstance(techniques, list), (
            "'techniques_applied' deve ser uma lista de técnicas"
        )
        assert len(techniques) >= 2, (
            f"Mínimo de 2 técnicas requeridas, encontradas: {len(techniques)}"
        )
        assert any("few-shot" in str(t).lower() for t in techniques), (
            "Few-shot Learning é obrigatória e não está em 'techniques_applied'"
        )

    def test_prompt_no_invalid_variables(self, prompt_data):
        """
        Verifica que {bug_report} é a ÚNICA variável de template no prompt.

        Qualquer outra chave simples {...} não escapada como {{...}} seria
        interpretada pelo LangChain como variável de entrada e quebraria a
        avaliação em runtime, já que o dataset só fornece 'bug_report'.
        """
        full_text = prompt_data["system_prompt"] + str(prompt_data.get("user_prompt", ""))

        # Captura {nome} ignorando chaves duplas {{...}}, que são o escape
        # do LangChain para chaves literais
        variables = set(re.findall(r"(?<!{){([a-zA-Z_0-9]+)}(?!})", full_text))

        assert variables == {"bug_report"}, (
            f"Variáveis de template inesperadas encontradas: {variables - {'bug_report'}} "
            "— escape chaves literais como '{{...}}' ou remova as variáveis extras"
        )

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
