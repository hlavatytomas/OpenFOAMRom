# -- Python script containing additional functions for OF_caseClass

def isFloat(val):
    """function to determine if val is float"""
    try:
        float(val)
        return True
    except:
        return False