local GameState = {}

-- Value to letter mapping for cards
local VALUE_TO_LETTER = {
    ["Ace"] = "A",
    ["King"] = "K",
    ["Queen"] = "Q",
    ["Jack"] = "J",
    ["10"] = "10",
    ["9"] = "9",
    ["8"] = "8",
    ["7"] = "7",
    ["6"] = "6",
    ["5"] = "5",
    ["4"] = "4",
    ["3"] = "3",
    ["2"] = "2"
}

-- Suit to letter mapping for cards
local SUIT_TO_LETTER = {
    ["Diamonds"] = "D",
    ["Hearts"] = "H",
    ["Clubs"] = "C",
    ["Spades"] = "S"
}

-- State processor for converting game state to vector
function GameState.process()
    local state = {}
    local idx = 1

    -- Add game state info
    state[idx] = G.GAME.chips
    idx = idx + 1

    state[idx] = G.GAME.current_round.hands_left
    idx = idx + 1
    
    state[idx] = G.GAME.current_round.discards_left
    idx = idx + 1

    -- Add whole hand to state
    for i, card in ipairs(G.hand.cards) do
        state[idx] = i
        state[idx + 1] = VALUE_TO_LETTER[card.base.value] .. SUIT_TO_LETTER[card.base.suit]
        idx = idx + 2
    end

    return state
end

-- Get formatted hand state for debugging
function GameState.getHandState()
    local hand_state = {
        whole_hand = {},
        selected_hand = {}
    }

    -- Add whole hand to state
    for i, card in ipairs(G.hand.cards) do
        table.insert(hand_state.whole_hand, {
            index = i,
            value = card.base.value,
            suit = card.base.suit
        })
    end

    -- Add selected hand to state
    for _, card in ipairs(G.hand.highlighted) do
        table.insert(hand_state.selected_hand, {
            value = card.base.value,
            suit = card.base.suit
        })
    end

    return hand_state
end

-- Check if game state is valid for processing
function GameState.isValid()
    return G and G.GAME and G.GAME.chips and G.GAME.current_round and 
           G.hand and G.hand.cards and #G.hand.cards > 0
end

-- Get current game state info for logging
function GameState.getStateInfo()
    if not GameState.isValid() then
        return "Invalid game state"
    end
    
    return string.format(
        "Chips: %d, Hands: %d, Discards: %d, Cards: %d",
        G.GAME.chips or 0,
        G.GAME.current_round.hands_left or 0,
        G.GAME.current_round.discards_left or 0,
        #G.hand.cards
    )
end

return GameState 