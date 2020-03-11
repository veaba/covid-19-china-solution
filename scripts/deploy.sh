#!/bin/bash
# 照着这个脚本抄的：https://github.com/peaceiris/actions-gh-pages/blob/master/entrypoint.sh#L39

set -e
# set -ex

function print_error() {
    echo -e "\e[31mERROR: ${1}\e[m"
}

function print_info() {
    echo -e "\e[36mINFO: ${1}\e[m"
}

function skip() {
    print_info "No changes detected, skipping deployment"
    exit 0
}

# check values
if [ -n "${EXTERNAL_REPOSITORY}" ]; then
    PUBLISH_REPOSITORY=${EXTERNAL_REPOSITORY}
else
    PUBLISH_REPOSITORY=${GITHUB_REPOSITORY}
fi
print_info "Deploy to ${PUBLISH_REPOSITORY}"

if [ -n "${ACTIONS_DEPLOY_KEY}" ]; then

    print_info "setup with ACTIONS_DEPLOY_KEY"

    if [ -n "${SCRIPT_MODE}" ]; then
        print_info "run as SCRIPT_MODE"
        SSH_DIR="${HOME}/.ssh"
    else
        SSH_DIR="/root/.ssh"
    fi
    mkdir "${SSH_DIR}"
    ssh-keyscan -t rsa github.com > "${SSH_DIR}/known_hosts"
    echo "${ACTIONS_DEPLOY_KEY}" > "${SSH_DIR}/id_rsa"
    chmod 400 "${SSH_DIR}/id_rsa"
    remote_repo="git@github.com:${PUBLISH_REPOSITORY}.git"
else
    print_error "not found ACTIONS_DEPLOY_KEY"
    exit 1
fi

if [ -z "${PUBLISH_BRANCH}" ]; then
    print_error "not found PUBLISH_BRANCH"
    exit 1
fi

if [ -z "${PUBLISH_DIR}" ]; then
    print_error "not found PUBLISH_DIR"
    exit 1
fi

remote_branch="${PUBLISH_BRANCH}"

local_dir="${HOME}/ghpages_${RANDOM}"
if git clone --depth=1 --single-branch --branch "${remote_branch}" "${remote_repo}" "${local_dir}"; then
    cd "${local_dir}"

    if [[ ${INPUT_KEEPFILES} == "true" ]]; then
        print_info "Keeping existing files: ${INPUT_KEEPFILES}"
    else
        git rm -r --ignore-unmatch '*'
    fi

    find "${GITHUB_WORKSPACE}/${PUBLISH_DIR}" -maxdepth 1 | \
        tail -n +2 | \
        xargs -I % cp -rf % "${local_dir}/"
    echo "进来吗？？"
else
    cd "${PUBLISH_DIR}"
    git init
    git checkout --orphan "${remote_branch}"
fi

echo "${local_dir}"

# push to publishing branch
git config user.name "${GITHUB_ACTOR}"
git config user.email "${GITHUB_ACTOR}@users.noreply.github.com"
git remote rm origin || true
git remote add origin "${remote_repo}"
git add --all

print_info "Allowing empty commits: ${INPUT_EMPTYCOMMITS}"
COMMIT_MESSAGE="【部署成功】: $(date -u) ${GITHUB_SHA}"
if [[ ${INPUT_EMPTYCOMMITS} == "false" ]]; then
    git commit -m "${COMMIT_MESSAGE}" || skip
else
    git commit --allow-empty -m "${COMMIT_MESSAGE}"
fi

git push origin "${remote_branch}"
print_info "${GITHUB_SHA} was successfully deployed"