# buyLotsOfFruit.py
# -----------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

"""
To run this script, type

  python buyLotsOfFruit.py
  
Once you have correctly implemented the buyLotsOfFruit function,
the script should produce the output:

Cost of [('apples', 2.0), ('pears', 3.0), ('limes', 4.0)] is 12.25
"""

fruitPrices = {'apples':2.00, 'oranges': 1.50, 'pears': 1.75,
              'limes':0.75, 'strawberries':1.00}


def buyLotsOfFruit(order_list):
    """
        orderList: List of (fruit, numPounds) tuples
            
    Returns cost of order
    """ 
    total_cost = 0.0
    for fruit, num_pounds in order_list:
        if fruitPrices[fruit]:
            total_cost += fruitPrices[fruit] * num_pounds
        else:
            print('Error: No known price for fruit: ', fruit)
            return None

    return total_cost


# Main Method    
if __name__ == '__main__':
    "This code runs when you invoke the script from the command line"
    orderList = [ ('apples', 2.0), ('pears', 3.0), ('limes', 4.0) ]
    print('Cost of', orderList, 'is', buyLotsOfFruit(orderList))