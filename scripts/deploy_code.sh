#!/bin/bash
# this script deploys main_res to a remote machine via rsync
# it excludes all personal directories except for the running user's
# it also excludes patterns listed in scripts/exclude_from_deploy.txt
# you can pass more rsync options at the end of the command without the dashes
# e.g. exclude=PATTERN dry-run delete
# (as opposed to --exclude=PATTERN --dry-run --delete)

set -e
usage () {
  echo "script usage: $(basename $0) -d <dest machine host name> [<additional rsync args (without dashes)>]" >&2
  }

while getopts 'd:h' OPTION; do
  case "$OPTION" in
    d) MACHINE="$OPTARG" ;;
    h) usage; exit 0 ;;
    ?) usage; exit 1 ;;
  esac
done

shift $(( OPTIND - 1 ))  # pop the first arg
RSYNC_FLAGS=""
for arg in "$@"; do
  RSYNC_FLAGS+=" --$arg";
done

RES_DIR=$( cd $( dirname "${BASH_SOURCE[0]}" )/..; pwd)
pushd $RES_DIR

PERSONAL_DIRS=$(echo Personal/* | sed "s|Personal/${USER}||I") # exclude all personal folders except mine (I for case insensitive)

# build exclusions
EXCLUSIONS=""
for P in $PERSONAL_DIRS; do
  EXCLUSIONS+=" --exclude ${P}/"
done



echo "additional rsync args: ${RSYNC_FLAGS}"
# --rsync-path hack to create the destination path on new machines
# trailing slash on source directory to copy hidden files
rsync -av --rsync-path="mkdir  -p ${RES_DIR} && rsync"  -e ssh ${RSYNC_FLAGS} --exclude-from "${RES_DIR}/scripts/exclude_from_deploy.txt" $EXCLUSIONS ./ $MACHINE:$RES_DIR/
popd

# ignore unsynced personal directories
ssh $MACHINE "cd $RES_DIR; git ls-files -z $PERSONAL_DIRS | xargs -0 git update-index --assume-unchanged"

echo success
echo

