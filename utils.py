def partition_num(num,workers):
    if num % workers == 0:
        return [num//workers]*workers
    else:
        return [num//workers]*workers+[num%workers]
    


        
