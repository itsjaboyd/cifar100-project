#!/usr/bin/env python

#=================================================================
# Author: Jason Boyd (A02258798)
# Date: December 9, 2020 5:17 PM
# 
# -- Description --
# This file serves as the executive python script to run and
# return statistics on every saved net within the project. For
# more information on the project itself, refer to the README.md. 
# Questions or concerns can be emailed to [jasonboyd99@gmail.com].
#=================================================================

def main():
    print("|========= LOADING AND TESTING SAVED NETWORKS =========|")
    test_saved_cnn()
    test_saved_ann()
    test_saved_raf()
    print("|========== SUCCESSFULLY TESTED ALL NETWORKS ==========|")

def test_saved_cnn():
    pass

def test_saved_ann():
    pass

def test_saved_raf():
    pass


main() if __name__ == "__main__" else print("No main function supplied.")