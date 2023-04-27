import numpy as np


def GreedySearch(SymbolSets, y_probs):
    """Greedy Search.

    Input
    -----
    SymbolSets: list
                all the symbols (the vocabulary without blank)

    y_probs: (# of symbols + 1, Seq_length, batch_size)
            Your batch size for part 1 will remain 1, but if you plan to use your
            implementation for part 2 you need to incorporate batch_size.

    Returns
    ------
    forward_path: str
                the corresponding compressed symbol sequence i.e. without blanks
                or repeated symbols.

    forward_prob: scalar (float)
                the forward probability of the greedy path

    """
    # Follow the pseudocode from lecture to complete greedy search :-)
    _,S,_=y_probs.shape
    forward_prob=1
    forward_path=[]
    symbols=['-']
    y_probs=np.squeeze(y_probs, axis=2)
    
    for i in range(len(SymbolSets)):
        symbols.append(SymbolSets[i])
    for i in range(S):
        forward_prob*=np.amax(y_probs[:,i])
        forward_path.append(symbols[np.argmax(y_probs[:,i])])
    
    path=''
    path=path+forward_path[0]
    for i in range(1,len(forward_path)):
        if (forward_path[i]=='-' or forward_path[i-1]==forward_path[i]):
            continue
        else:
            path+=forward_path[i]
             
    
        
    return path, forward_prob
    #raise NotImplementedError


##############################################################################


def BeamSearch(SymbolSets, y_probs, BeamWidth):
    """Beam Search.

    Input
    -----
    SymbolSets: list
                all the symbols (the vocabulary without blank)

    y_probs: (# of symbols + 1, Seq_length, batch_size)
            Your batch size for part 1 will remain 1, but if you plan to use your
            implementation for part 2 you need to incorporate batch_size.

    BeamWidth: int
                Width of the beam.

    Return
    ------
    bestPath: str
            the symbol sequence with the best path score (forward probability)

    mergedPathScores: dictionary
                        all the final merged paths with their scores.

    """
    # Follow the pseudocode from lecture to complete beam search :-)
    PathScore, BlankPathScore={},{}
    
    def InitializePaths(SymbolSet, y):
        InitialBlankPathScore = {}
        InitialPathScore = {}
        # First push the blank into a path-ending-with-blank stack. No symbol has been invoked yet
        path = ""
        InitialBlankPathScore[path]=y[0] # Score of blank at t=1
        InitialPathsWithFinalBlank = set(path)
        # Push rest of the symbols into a path-ending-with-symbol stack
        InitialPathsWithFinalSymbol = set()
        for c in range(len(SymbolSet)): # This is the entire symbol set, without the blank
            path = SymbolSet[c]
            InitialPathScore[path] = y[c+1] # Score of symbol c at t=1
            InitialPathsWithFinalSymbol.add(path) # Set addition

        return InitialPathsWithFinalBlank, InitialPathsWithFinalSymbol, InitialBlankPathScore, InitialPathScore
    
    def ExtendWithBlank(PathsWithTerminalBlank, PathsWithTerminalSymbol, y):
        UpdatedPathsWithTerminalBlank = set()
        UpdatedBlankPathScore = {}
        # First work on paths with terminal blanks
        #(This represents transitions along horizontal trellis edges for blanks)
        for path in PathsWithTerminalBlank:
            # Repeating a blank doesn’t change the symbol sequence
            UpdatedPathsWithTerminalBlank.add(path) # Set addition
            UpdatedBlankPathScore[path] = BlankPathScore[path]*y[0]
        
        # Then extend paths with terminal symbols by blanks
        for path in PathsWithTerminalSymbol:
            # If there is already an equivalent string in UpdatesPathsWithTerminalBlank
            # simply add the score. If not create a new entry
            if path in UpdatedPathsWithTerminalBlank:
                UpdatedBlankPathScore[path] += PathScore[path]* y[0]
            else:
                UpdatedPathsWithTerminalBlank.add(path) # Set addition
                UpdatedBlankPathScore[path] = PathScore[path] * y[0]

        return UpdatedPathsWithTerminalBlank, UpdatedBlankPathScore
    
    def ExtendWithSymbol(PathsWithTerminalBlank, PathsWithTerminalSymbol, SymbolSet, y):
        UpdatedPathsWithTerminalSymbol = set()
        UpdatedPathScore = {}
        # First extend the paths terminating in blanks. This will always create a new sequence
        for path in PathsWithTerminalBlank:
            for c in range(len(SymbolSet)): # SymbolSet does not include blanks
                newpath = path + SymbolSet[c] # Concatenation
                UpdatedPathsWithTerminalSymbol.add(newpath) # Set addition
                UpdatedPathScore[newpath] = BlankPathScore[path] * y[c+1]

        # Next work on paths with terminal symbols
        for path in PathsWithTerminalSymbol:
            # Extend the path with every symbol other than blank
            for c in range(len(SymbolSet)): # SymbolSet does not include blanks
                if(SymbolSet[c] == path[-1]):
                    newpath = path
                else:
                    newpath=path + SymbolSet[c]
                  
                if newpath in UpdatedPathsWithTerminalSymbol: # Already in list, merge paths
                    UpdatedPathScore[newpath] += PathScore[path] * y[c+1]
                else: # Create new path
                    UpdatedPathsWithTerminalSymbol.add(newpath) # Set addition
                    UpdatedPathScore[newpath] = PathScore[path] * y[c+1]

        return UpdatedPathsWithTerminalSymbol, UpdatedPathScore
    
    def Prune(PathsWithTerminalBlank, PathsWithTerminalSymbol, BlankPathScore, PathScore, BeamWidth):
            PrunedBlankPathScore = {}
            PrunedPathScore = {}
            scorelist=[]
            # First gather all the relevant scores
            i = 0
            for p in PathsWithTerminalBlank:
                scorelist.append(BlankPathScore[p])
                i+=1

            for p in PathsWithTerminalSymbol:
                scorelist.append(PathScore[p])
                i+=1

            # Sort and find cutoff score that retains exactly BeamWidth paths
            scorelist.sort(reverse=True) # In decreasing order
            if(BeamWidth < len(scorelist)):
                cutoff = scorelist[BeamWidth]
            else:
                cutoff=scorelist[-1]
            PrunedPathsWithTerminalBlank = set()
            for p in PathsWithTerminalBlank:
                if BlankPathScore[p] > cutoff:
                    PrunedPathsWithTerminalBlank.add(p) # Set addition
                    PrunedBlankPathScore[p] = BlankPathScore[p]

            PrunedPathsWithTerminalSymbol = set()
            for p in PathsWithTerminalSymbol:
                if PathScore[p] > cutoff:
                    PrunedPathsWithTerminalSymbol.add(p) # Set addition
                    PrunedPathScore[p] = PathScore[p]

            return PrunedPathsWithTerminalBlank, PrunedPathsWithTerminalSymbol, PrunedBlankPathScore, PrunedPathScore
    
    def MergeIdenticalPaths(PathsWithTerminalBlank, BlankPathScore,PathsWithTerminalSymbol, PathScore):
        # All paths with terminal symbols will remain
        MergedPaths = PathsWithTerminalSymbol
        FinalPathScore = PathScore
        # Paths with terminal blanks will contribute scores to existing identical paths from
        # PathsWithTerminalSymbol if present, or be included in the final set, otherwise
        for p in PathsWithTerminalBlank:
            if p in MergedPaths:
                FinalPathScore[p] += BlankPathScore[p]
            else:
                MergedPaths+=p # Set addition
                FinalPathScore[p] = BlankPathScore[p]
        return MergedPaths, FinalPathScore
    
    # First time instant: Initialize paths with each of the symbols,
    # including blank, using score at time t=1
    #y_probs=np.squeeze(y_probs, axis=2)
    NewPathsWithTerminalBlank, NewPathsWithTerminalSymbol, NewBlankPathScore, NewPathScore = \
        InitializePaths(SymbolSets, y_probs[:,0,0])
    # Subsequent time steps
    for t in range(1,y_probs.shape[1]):
        # Prune the collection down to the BeamWidth
        PathsWithTerminalBlank, PathsWithTerminalSymbol, BlankPathScore, PathScore = \
            Prune(NewPathsWithTerminalBlank, NewPathsWithTerminalSymbol,NewBlankPathScore, NewPathScore,\
                  BeamWidth)
        # First extend paths by a blank
        NewPathsWithTerminalBlank, NewBlankPathScore = ExtendWithBlank(PathsWithTerminalBlank,\
                                                                       PathsWithTerminalSymbol, \
                                                                           y_probs[:,t,0])
        # Next extend paths by a symbol
        NewPathsWithTerminalSymbol, NewPathScore = ExtendWithSymbol(PathsWithTerminalBlank,\
                                                                    PathsWithTerminalSymbol, SymbolSets,\
                                                                        y_probs[:,t,0])

    # Merge identical paths differing only by the final blank
    MergedPaths, FinalPathScore = MergeIdenticalPaths(NewPathsWithTerminalBlank, NewBlankPathScore, \
                                                      NewPathsWithTerminalSymbol, NewPathScore)
    # Pick best path
    BestPath =max(FinalPathScore, key=FinalPathScore.get) # Find the path with the best score
    
    return BestPath, FinalPathScore
    #raise NotImplementedError
