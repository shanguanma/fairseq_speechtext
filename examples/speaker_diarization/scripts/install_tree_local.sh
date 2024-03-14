#!/bin/sh
PREFIX="$HOME/.local/"

install_tree() {
    # The project page of linux "tree" command is located at http://mama.indstate.edu/users/ice/tree
    TMP_TREE_DIR="/tmp/$USER/tree"; mkdir -p $TMP_TREE_DIR

    wget -nc -O $TMP_TREE_DIR/unix-tree-2.1.1.tar.bz2 "https://gitlab.com/OldManProgrammer/unix-tree/-/archive/2.1.1/unix-tree-2.1.1.tar.bz2"
    tar -jxvf  $TMP_TREE_DIR/unix-tree-2.1.1.tar.bz2 -C $TMP_TREE_DIR/

    cd $TMP_TREE_DIR/unix-tree-2.1.1
    make
    mkdir -p "$PREFIX/bin"
    mv -v tree "$PREFIX/bin"
    # Maybe $PREFIX/bin was added to $PATH, if you run the wookayin/dotfiles script
    # echo "export PATH=$PREFIX/bin:$PATH"
}

install_tree
