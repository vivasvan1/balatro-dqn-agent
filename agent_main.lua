Lovely = require("lovely")
HandObserver = {}

Agent = {
    -- Private state
    _is_file_written = false,
    _ctrl_g_pressed = false,
}

-- Helper function to serialize tables
local function serializeTable(tbl, visited, path)
    visited = visited or {}
    path = path or ""
    if visited[tbl] then 
        return "\"[Circular Reference at " .. path .. "]\"" 
    end
    visited[tbl] = true

    local result = "{"
    local first = true
    for k, v in pairs(tbl) do
        if not first then result = result .. "," end
        first = false

        -- Handle key
        if type(k) == "string" then
            result = result .. "\"" .. k .. "\":"
        else
            result = result .. "\"" .. tostring(k) .. "\":"
        end

        -- Handle value
        if type(v) == "table" then
            local new_path = path .. (path ~= "" and "." or "") .. tostring(k)
            result = result .. serializeTable(v, visited, new_path)
        elseif type(v) == "string" then
            result = result .. "\"" .. v:gsub("\"", "\\\"") .. "\""
        elseif type(v) == "function" then
            -- Use debug.getinfo to get function info
            local info = debug.getinfo(v, "uS")
            local args = {}
            if info and info.nparams then
                for i = 1, info.nparams do
                    local name = debug.getlocal(v, i)
                    if name then table.insert(args, name) end
                end
            end
            local argstr = table.concat(args, ", ")
            if info and info.what == "C" then
                result = result .. string.format("\"[C Function]\"")
            else
                result = result .. string.format("\"[Function(args: %s)]\"", argstr)
            end
        elseif type(v) == "userdata" or type(v) == "thread" then
            result = result .. "\"[" .. type(v) .. "]\""
        else
            result = result .. tostring(v)
        end
    end
    result = result .. "}"
    return result
end

-- Helper function to save game state to JSON
local function saveGameStateToJson(data, filename)
    local json_path = Lovely.mod_dir .. "/balatro_agent/" .. filename
    local file = nativefs.newFile(json_path)
    local json_data = serializeTable(data)
    local success, error_msg = file:open("w")
    if success then
        file:write(json_data)
        file:close()
        print("Game state saved to: " .. json_path)
        return true
    else
        print("Failed to create JSON file: " .. (error_msg or "unknown error"))
        return false
    end
end

-- Helper function to check Ctrl+G
function Agent._check_ctrl_g()
    local ctrl = love.keyboard.isDown("lctrl") or love.keyboard.isDown("rctrl")
    local g = love.keyboard.isDown("g")
    
    if ctrl and g and not Agent._ctrl_g_pressed then
        print("Ctrl+G detected!")
    end
    
    if ctrl and g then
        if not Agent._ctrl_g_pressed then
            Agent._ctrl_g_pressed = true
            return true
        end
    else
        Agent._ctrl_g_pressed = false
    end
    return false
end

-- Helper function to select cards by indices
function Agent._select_hand_cards_by_indices(indices)
    if not G or not G.hand or not G.hand.cards then
        print("No valid hand found.")
        return
    end

    -- First unhighlight all cards
    G.hand:unhighlight_all()

    -- Validate indices
    for _, idx in ipairs(indices) do
        if type(idx) ~= "number" or idx < 1 or idx > #G.hand.cards then
            print("Invalid index: " .. tostring(idx))
            return
        end
    end

    -- Print whole hand
    print("\nWhole Hand:")
    for i, card in ipairs(G.hand.cards) do
        print(string.format("Card %d: %s", i, card.base.value .. card.base.suit))
    end

    -- Highlight cards at specified indices
    for _, idx in ipairs(indices) do
        local card = G.hand.cards[idx]
        -- Call the highlight method on the card
        card:highlight(true)
        -- Add to highlighted array - makes sure internal tracking is correct
        table.insert(G.hand.highlighted, card)
    end

    -- If we're selecting a hand, make sure to parse the highlighted cards
    if G.STATE == G.STATES.SELECTING_HAND then
        G.hand:parse_highlighted()
    end

    -- Print selected hand
    print("\nSelected Hand:")
    for _, card in ipairs(G.hand.highlighted) do
        print(string.format("Selected: %s", card.base.value .. card.base.suit))
    end

    -- Save hand state to JSON
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

    -- Save to JSON file
    saveGameStateToJson(hand_state, "hand_state.json")

    print("\nHighlighted " .. #indices .. " cards in hand.")
end

-- Helper function to select random hand cards
function Agent._select_random_hand_cards()
    if not G or not G.hand or not G.hand.cards or #G.hand.cards < 5 then
        print("Not enough cards in hand to select 5.")
        return
    end

    -- Print current score
    if G.GAME and G.GAME.current_round and G.GAME.current_round.current_hand then
        print("Current Score:")
        print("Current Hand Chips: " .. (G.GAME.current_round.current_hand.chips or 0))
        print("Current Hand Multiplier: " .. (G.GAME.current_round.current_hand.mult or 0))
        print("Current Hand Total: " .. ((G.GAME.current_round.current_hand.chips or 0) * (G.GAME.current_round.current_hand.mult or 0)))
        print("Current Round: " .. (G.GAME.round or 0))
        print("Current Ante: " .. (G.GAME.round_resets.ante or 0))
        print("Discards Left: " .. (G.GAME.current_round.discards_left or 0))
        print("Hands Left: " .. (G.GAME.current_round.hands_left or 0))
        print("Round Total Score: " .. (G.GAME.current_round.chips or 0))
        print("Current Chips: " .. (G.GAME.chips or 0))

    end


    -- Generate random indices
    local indices = {}
    for i = 1, #G.hand.cards do table.insert(indices, i) end
    for i = #indices, 2, -1 do
        local j = math.random(i)
        indices[i], indices[j] = indices[j], indices[i]
    end

    -- Select first 5 random cards
    local selected_indices = {}
    local max_cards = math.min(5, #G.hand.cards)
    for i = 1, max_cards do
        table.insert(selected_indices, indices[i])
    end

    -- Use the new function to select the cards
    Agent._select_hand_cards_by_indices(selected_indices)

    -- G.FUNCS.discard_cards_from_highlighted()
end

-- Main initialization function
function Agent.init()

    print("Agent.init() called")

    -- Save initial game state
    -- saveGameStateToJson(G, "G.json")
    -- saveGameStateToJson(G.FUNCS, "G.FUNCS.json")

    -- Load and initialize HandObserver
    local main_logic_path = Lovely.mod_dir .. "hand_observer_main.lua"
    if nativefs.getInfo(main_logic_path) then
        assert(load(nativefs.read(main_logic_path)))()
        if HandObserver and HandObserver.init then
            HandObserver.init()
            print("HandObserver initialized. Ensure lovely.toml patches an update loop to call HandObserver.observe_hand().")
        else
            print("Failed to initialize HandObserver: HandObserver.init not found after loading main logic.")
        end
    else
        print("Error: Could not find HandObserver/hand_observer_main.lua at path: " .. main_logic_path)
    end
end

-- Main update function
function Agent.update(dt)
    -- Handle Ctrl+G keypress
    if Agent._check_ctrl_g() then
        print("Selecting random cards due to Ctrl+G press")
        print(G.STATE)
        Agent._select_random_hand_cards()
    end

    

    -- Save game stages state once
    -- if G.STAGE == G.STAGES.RUN and not G.SETTINGS.paused and not G.OVERLAY_MENU and not Agent._is_file_written then
    --     Agent._is_file_written = true
    --     print(G.STAGES)
    --     if type(G.STAGES) == "table" then
    --         saveGameStateToJson(G.STAGES, "G.STAGES.json")
    --     else
    --         print("Error: G.STAGES is not a table and cannot be serialized.")
    --     end
    -- end
end

print("Agent module loaded, Agent.init() is ready.")
