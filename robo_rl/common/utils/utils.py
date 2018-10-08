def heading_decorator(bottom=False, top=False, print_req=False):
    deco = '\n-------------------------------------------'
    if top is True:
        if print_req is False:
            return deco + '\n'
        else:
            print(deco + '\n')
    if bottom is True:
        if print_req is False:
            return deco
        else:
            print(deco)


def print_heading(heading, ):
    print(heading_decorator(top=True) + heading + heading_decorator(bottom=True))
