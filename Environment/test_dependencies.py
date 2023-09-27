"""
    Quick script to test if any project dependencies are missing
"""
if __name__ == '__main__':
    try:
        import torch
        import torchvision
        import numpy

        print("All dependecies are satisfied!")
    except ModuleNotFoundError as e:
        print("Some dependencies missed:{}".format(e))
