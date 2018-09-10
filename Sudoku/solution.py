rows = 'ABCDEFGHI'
cols = '123456789'

assignments = []

def assign_value(values, box, value):
    """
    Please use this function to update your values dictionary!
    Assigns a value to a given box. If it updates the board record it.
    """
    values[box] = value
    if len(value) == 1:
        assignments.append(values.copy())
    return values

def cross(a, b):
    return [s+t for s in a for t in b]

boxes = cross(rows, cols)

row_units = [cross(r, cols) for r in rows]

column_units = [cross(rows, c) for c in cols]
leading_diagonal_units = [[rows[i]+cols[i]for i in range(0,9)]]
minor_diagonal_units = [[rows[-i]+cols[i-1]for i in range(9,0,-1)]]
square_units = [cross(rs, cs) for rs in ('ABC','DEF','GHI') for cs in ('123','456','789')]
unitlist = row_units + column_units +leading_diagonal_units + minor_diagonal_units + square_units
units = dict((s, [u for u in unitlist if s in u]) for s in boxes)
peers = dict((s, set(sum(units[s],[]))-set([s])) for s in boxes)



def display(values):
    """
    Display the values as a 2-D grid.
    Input: The sudoku in dictionary form
    Output: None
    """
    width = 1+max(len(values[s]) for s in boxes)
    line = '+'.join(['-'*(width*3)]*3)
    for r in rows:
        print(''.join(values[r+c].center(width)+('|' if c in '36' else '')
                      for c in cols))
        if r in 'CF': print(line)
    return

def grid_values(grid):
    """Convert grid string into {<box>: <value>} dict with '.' value for empties.

    Args:
        grid: Sudoku grid in string form, 81 characters long
    Returns:
        Sudoku grid in dictionary form:
        - keys: Box labels, e.g. 'A1'
        - values: Value in corresponding box, e.g. '8', or '.' if it is empty.
    """
    assert len(grid) == 81, "Input grid must be a string of length 81 (9x9)"
    values=[]
    total_digits = '123456789'
    for c in grid:
        if c=='.':
            values.append(total_digits)
        elif c in total_digits:
            values.append(c)

    assert len(values) == 81, "Input grid must be a string of length 81 (9x9)"
    return dict(zip(boxes,values))

def eliminate(values):
    """Eliminate values from peers of each box with a single value.

    Go through all the boxes, and whenever there is a box with a single value,
    eliminate this value from the set of values of all its peers.

    Args:
        values: Sudoku in dictionary form.
    Returns:
        Resulting Sudoku in dictionary form after eliminating values.
    """
    assert len(values) == 81, "Input grid must be a string of length 81 (9x9)"
    solved_grid_values = [box for box in values.keys() if len(values[box]) == 1]
    for box in solved_grid_values:
        digit = values[box]
        if len(digit) == 1:
            for peer in peers[box]:
                new_digit = values[peer].replace(digit,'')
                values = assign_value(values,peer,new_digit)
                #values[peer] = new_digit
    return values

def only_choice(values):
    """Finalize all values that are the only choice for a unit.

    Go through all the units, and whenever there is a unit with a value
    that only fits in one box, assign the value to this box.

    Input: Sudoku in dictionary form.
    Output: Resulting Sudoku in dictionary form after filling in only choices.
    """
    assert len(values) == 81, "Input grid must be a string of length 81 (9x9)"
    for unit in unitlist:
        for digit in '123456789':
            dplaces = [box for box in unit if digit in values[box]]
            if len(dplaces) == 1:
                #values = assign_value(values, values[dplaces[0]], digit)
                box = dplaces[0]
                values = assign_value(values,box,digit)
                #values[dplaces[0]] = digit
    return values

def reduce_puzzle(values):
    stalled = False
    while not stalled:
        # Check how many boxes have a determined value
        solved_values_before = len([box for box in values.keys() if len(values[box]) == 1])

        values = eliminate(values)   

        values = only_choice(values)

        values = naked_twins(values)

        # Check how many boxes have a determined value, to compare
        solved_values_after = len([box for box in values.keys() if len(values[box]) == 1])
        # If no new values were added, stop the loop.
        stalled = solved_values_before == solved_values_after
        # Sanity check, return False if there is a box with zero available values:
        if len([box for box in values.keys() if len(values[box]) == 0]):
            return False
    return values

def search(values):
    "Using depth-first search and propagation, try all possible values."
    values = reduce_puzzle(values)
    if values is False:
        return False ## Failed earlier
    if all(len(values[s]) == 1 for s in boxes):
        return values ## Solved!
    # Choose one of the unfilled squares with the fewest possibilities
    result_tuples=[]
    for s in boxes:
        if (len(values[s]) > 1):
            tuple = (len(values[s]),s)
            result_tuples.append(tuple)
            n,s = min(result_tuples)
    #n,s = min((len(values[s]), s) for s in boxes if len(values[s]) > 1)
    # Now use recurrence to solve each one of the resulting sudokus, and
    for value in values[s]:
        new_sudoku = values.copy()
        new_sudoku[s] = value
        attempt = search(new_sudoku)
        if attempt:
            return attempt

def naked_twins(values):
    """Eliminate values using the naked twins strategy.
    Args:
        values(dict): a dictionary of the form {'box_name': '123456789', ...}

    Returns:
        the values dictionary with the naked twins eliminated from peers.
    """
    # Find all instances of naked twins
    probable_twins = [box for box in values.keys() if len(values[box]) == 2]
    # Collect boxes that have the same elements
    naked_twins = [[box1, box2] for box1 in probable_twins for box2 in peers[box1] if set(values[box1]) == set(values[box2])]

    # For each pair of naked twins,
    for i in range(len(naked_twins)):
        box1 = naked_twins[i][0]
        box2 = naked_twins[i][1]
        # 1- compute intersection of peers
        peers1 = set(peers[box1])
        peers2 = set(peers[box2])
        peers_int = peers1 & peers2
        # 2- Delete the two digits in naked twins from all common peers.
        for peer_val in peers_int:
            if len(values[peer_val]) > 2:
                for rm_val in values[box1]:
                    values = assign_value(values, peer_val, values[peer_val].replace(rm_val, ''))
    return values

def solve(grid):
    """
    Find the solution to a Sudoku grid.
    Args:
        grid(string): a string representing a sudoku grid.
            Example: '2.............62....1....7...6..8...3...9...7...6..4...4....8....52.............3'
    Returns:
        The dictionary representation of the final sudoku grid. False if no solution exists.
    """
    values = grid_values(grid)
    #result = eliminate(values)
    #result = only_choice(values)
    #result = reduce_puzzle(values)
    result = search(values)
    if result:
        return result
    else:
        return False

if __name__ == '__main__':
    diag_sudoku_grid = '2.............62....1....7...6..8...3...9...7...6..4...4....8....52.............3'
    #diag_sudoku_grid = '..3.2.6..9..3.5..1..18.64....81.29..7.......8..67.82....26.95..8..2.3..9..5.1.3..'
 
    display(solve(diag_sudoku_grid))

    try:
        from visualize import visualize_assignments
        visualize_assignments(assignments)

    except SystemExit:
        pass
    except:
        print('We could not visualize your board due to a pygame issue. Not a problem! It is not a requirement.')
