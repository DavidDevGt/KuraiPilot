#!/usr/bin/env bash
# Configura el repo de GitHub con las prácticas de CONTRIBUTING.md.
# Idempotente: correrlo dos veces no rompe nada.
#
# Requisitos: gh CLI autenticado con permisos de admin sobre el repo.
#   sudo apt install gh && gh auth login
# Uso: ./tools/setup_github.sh

set -euo pipefail

REPO="DavidDevGt/KuraiPilot"

echo "==> Verificando gh"
gh auth status >/dev/null

echo "==> Settings del repo: squash-only, auto-delete de ramas, sin wiki/projects"
gh api -X PATCH "repos/${REPO}" \
  -f description="Conversor local-first de video a video ASCII renderizado — determinista por defecto, IA por elección" \
  -F allow_squash_merge=true \
  -F allow_merge_commit=false \
  -F allow_rebase_merge=false \
  -F delete_branch_on_merge=true \
  -F squash_merge_commit_title=PR_TITLE \
  -F squash_merge_commit_message=PR_BODY \
  -F has_wiki=false \
  -F has_projects=false \
  -F allow_update_branch=true >/dev/null
echo "    ✓"

echo "==> Branch protection en main (CI requerido, 1 review + code owners, lineal)"
# Los contexts deben coincidir con los nombres de job de ci.yml
gh api -X PUT "repos/${REPO}/branches/main/protection" \
  --input - >/dev/null <<'JSON'
{
  "required_status_checks": {
    "strict": true,
    "contexts": [
      "lint + format + types",
      "pytest (py3.12)",
      "pytest (py3.13)"
    ]
  },
  "enforce_admins": false,
  "required_pull_request_reviews": {
    "required_approving_review_count": 1,
    "require_code_owner_reviews": true,
    "dismiss_stale_reviews": true
  },
  "restrictions": null,
  "required_linear_history": true,
  "allow_force_pushes": false,
  "allow_deletions": false,
  "required_conversation_resolution": true
}
JSON
echo "    ✓"

echo "==> Labels del flujo de trabajo"
create_label() { # nombre color descripción
  gh api -X POST "repos/${REPO}/labels" \
    -f name="$1" -f color="$2" -f description="$3" >/dev/null 2>&1 \
    || gh api -X PATCH "repos/${REPO}/labels/$(printf %s "$1" | sed 's/ /%20/g')" \
         -f color="$2" -f description="$3" >/dev/null
}
create_label "fase-0"          "0e8a16" "Core determinista (docs/07)"
create_label "fase-0.5"        "5db85b" "Preview y terminal live"
create_label "fase-1"          "fbca04" "Saliencia — gate A/B"
create_label "fase-2"          "d93f0b" "Alta fidelidad (CNN, FS, flow)"
create_label "fase-3"          "b60205" "Scene Analyst y exports"
create_label "hot-path"        "e11d21" "Toca etapas 2-8: exige bench de máquina de referencia"
create_label "adr"             "5319e7" "Propone o supersede una decisión estructural"
create_label "golden"          "bfdadc" "Modifica golden files: exige justificación de algoritmo"
create_label "good first issue" "7057ff" "Acotado y sin GPU: ideal para arrancar"
echo "    ✓"

echo
echo "Listo. Verificá en: https://github.com/${REPO}/settings/branches"
