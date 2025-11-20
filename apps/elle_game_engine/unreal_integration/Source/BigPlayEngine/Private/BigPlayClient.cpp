// BigPlayClient.cpp
// BigPlay Engine - Implementation
// Version: 1.0.0

#include "BigPlayClient.h"
#include "HttpModule.h"
#include "Interfaces/IHttpResponse.h"
#include "Dom/JsonObject.h"
#include "Serialization/JsonReader.h"
#include "Serialization/JsonSerializer.h"
#include "Serialization/JsonWriter.h"

void UBigPlayClient::Initialize(const FString& InBaseURL)
{
    BaseURL = InBaseURL;
    UE_LOG(LogTemp, Log, TEXT("[BigPlay] Initialized with base URL: %s"), *BaseURL);
}

void UBigPlayClient::GetNPCDialogue(
    const FString& NPCId,
    const FString& SceneId,
    const FString& PlayerMessage,
    FOnActionReceived OnSuccess,
    FOnRequestFailed OnFailure)
{
    // Build simplified game state for NPC dialogue
    FBigPlayGameState GameState;
    GameState.SceneId = SceneId;

    // Create NPC
    FBigPlayNPC NPC;
    NPC.Id = NPCId;
    NPC.Name = NPCId; // Simplified - game should provide real name
    NPC.Role = TEXT("npc");
    NPC.Location = SceneId;
    GameState.NPCs.Add(NPC);

    // Player intent
    FBigPlayPlayerIntent Intent;
    Intent.Type = EBigPlayIntentType::TalkToNPC;
    Intent.TargetNPCId = NPCId;
    Intent.RawInput = PlayerMessage;

    SendGameAction(GameState, Intent, OnSuccess, OnFailure);
}

void UBigPlayClient::GetHint(
    const FString& SceneId,
    const FString& QuestStage,
    FOnActionReceived OnSuccess,
    FOnRequestFailed OnFailure)
{
    FBigPlayGameState GameState;
    GameState.SceneId = SceneId;
    GameState.Player.QuestStage = QuestStage;

    FBigPlayPlayerIntent Intent;
    Intent.Type = EBigPlayIntentType::RequestHint;

    SendGameAction(GameState, Intent, OnSuccess, OnFailure);
}

void UBigPlayClient::GetWorldReaction(
    const FString& SceneId,
    const FString& TimeOfDay,
    const FString& Weather,
    FOnActionReceived OnSuccess,
    FOnRequestFailed OnFailure)
{
    FBigPlayGameState GameState;
    GameState.SceneId = SceneId;
    GameState.World.TimeOfDay = TimeOfDay;
    GameState.World.Weather = Weather;

    FBigPlayPlayerIntent Intent;
    Intent.Type = EBigPlayIntentType::EnterScene;

    SendGameAction(GameState, Intent, OnSuccess, OnFailure);
}

void UBigPlayClient::SendGameAction(
    const FBigPlayGameState& GameState,
    const FBigPlayPlayerIntent& PlayerIntent,
    FOnActionReceived OnSuccess,
    FOnRequestFailed OnFailure)
{
    // Build JSON payload
    TSharedPtr<FJsonObject> RootObject = MakeShareable(new FJsonObject);
    RootObject->SetObjectField(TEXT("game_state"), BuildGameStateJson(GameState));
    RootObject->SetObjectField(TEXT("player_intent"), BuildPlayerIntentJson(PlayerIntent));

    SendRequest(TEXT("/elle/game/action"), TEXT("POST"), RootObject, OnSuccess, OnFailure);
}

void UBigPlayClient::CheckHealth(
    FOnActionReceived OnSuccess,
    FOnRequestFailed OnFailure)
{
    SendRequest(TEXT("/health"), TEXT("GET"), nullptr, OnSuccess, OnFailure);
}

void UBigPlayClient::SendRequest(
    const FString& Endpoint,
    const FString& Method,
    TSharedPtr<FJsonObject> JsonPayload,
    FOnActionReceived OnSuccess,
    FOnRequestFailed OnFailure)
{
    // Create HTTP request
    TSharedRef<IHttpRequest> HttpRequest = FHttpModule::Get().CreateRequest();
    HttpRequest->SetVerb(Method);
    HttpRequest->SetURL(BaseURL + Endpoint);
    HttpRequest->SetHeader(TEXT("Content-Type"), TEXT("application/json"));
    HttpRequest->SetTimeout(Timeout);

    // Serialize JSON payload
    if (JsonPayload.IsValid())
    {
        FString OutputString;
        TSharedRef<TJsonWriter<>> Writer = TJsonWriterFactory<>::Create(&OutputString);
        FJsonSerializer::Serialize(JsonPayload.ToSharedRef(), Writer);
        HttpRequest->SetContentAsString(OutputString);

        UE_LOG(LogTemp, Verbose, TEXT("[BigPlay] Request: %s %s\nPayload: %s"),
            *Method, *Endpoint, *OutputString);
    }
    else
    {
        UE_LOG(LogTemp, Verbose, TEXT("[BigPlay] Request: %s %s"), *Method, *Endpoint);
    }

    // Bind response callback
    HttpRequest->OnProcessRequestComplete().BindUObject(
        this,
        &UBigPlayClient::OnResponseReceived,
        OnSuccess,
        OnFailure
    );

    // Send request
    if (!HttpRequest->ProcessRequest())
    {
        OnFailure.ExecuteIfBound(TEXT("Failed to send HTTP request"));
    }
}

void UBigPlayClient::OnResponseReceived(
    FHttpRequestPtr Request,
    FHttpResponsePtr Response,
    bool bWasSuccessful,
    FOnActionReceived OnSuccess,
    FOnRequestFailed OnFailure)
{
    if (!bWasSuccessful || !Response.IsValid())
    {
        FString ErrorMsg = FString::Printf(
            TEXT("HTTP request failed: %s"),
            Request.IsValid() ? *Request->GetURL() : TEXT("Unknown")
        );
        UE_LOG(LogTemp, Error, TEXT("[BigPlay] %s"), *ErrorMsg);
        OnFailure.ExecuteIfBound(ErrorMsg);
        return;
    }

    int32 ResponseCode = Response->GetResponseCode();
    FString ResponseBody = Response->GetContentAsString();

    UE_LOG(LogTemp, Verbose, TEXT("[BigPlay] Response (%d): %s"), ResponseCode, *ResponseBody);

    if (ResponseCode < 200 || ResponseCode >= 300)
    {
        FString ErrorMsg = FString::Printf(
            TEXT("HTTP %d: %s"),
            ResponseCode,
            *ResponseBody
        );
        UE_LOG(LogTemp, Error, TEXT("[BigPlay] %s"), *ErrorMsg);
        OnFailure.ExecuteIfBound(ErrorMsg);
        return;
    }

    // Parse JSON response
    TSharedPtr<FJsonObject> JsonObject;
    TSharedRef<TJsonReader<>> Reader = TJsonReaderFactory<>::Create(ResponseBody);

    if (!FJsonSerializer::Deserialize(Reader, JsonObject) || !JsonObject.IsValid())
    {
        FString ErrorMsg = TEXT("Failed to parse JSON response");
        UE_LOG(LogTemp, Error, TEXT("[BigPlay] %s"), *ErrorMsg);
        OnFailure.ExecuteIfBound(ErrorMsg);
        return;
    }

    // Parse action
    FBigPlayAction Action = ParseActionResponse(JsonObject);
    OnSuccess.ExecuteIfBound(Action);
}

TSharedPtr<FJsonObject> UBigPlayClient::BuildGameStateJson(const FBigPlayGameState& GameState)
{
    TSharedPtr<FJsonObject> JsonObject = MakeShareable(new FJsonObject);

    JsonObject->SetStringField(TEXT("scene_id"), GameState.SceneId);

    // NPCs array
    TArray<TSharedPtr<FJsonValue>> NPCsArray;
    for (const FBigPlayNPC& NPC : GameState.NPCs)
    {
        NPCsArray.Add(MakeShareable(new FJsonValueObject(BuildNPCJson(NPC))));
    }
    JsonObject->SetArrayField(TEXT("npcs"), NPCsArray);

    // Player
    TSharedPtr<FJsonObject> PlayerJson = MakeShareable(new FJsonObject);
    PlayerJson->SetStringField(TEXT("name"), GameState.Player.Name);
    PlayerJson->SetStringField(TEXT("location"), GameState.Player.Location);
    if (!GameState.Player.QuestStage.IsEmpty())
    {
        PlayerJson->SetStringField(TEXT("quest_stage"), GameState.Player.QuestStage);
    }
    if (!GameState.Player.Reputation.IsEmpty())
    {
        PlayerJson->SetStringField(TEXT("reputation"), GameState.Player.Reputation);
    }

    // Traits
    TSharedPtr<FJsonObject> TraitsJson = MakeShareable(new FJsonObject);
    for (const auto& Pair : GameState.Player.Traits)
    {
        TraitsJson->SetNumberField(Pair.Key, Pair.Value);
    }
    PlayerJson->SetObjectField(TEXT("traits"), TraitsJson);

    // Inventory tags
    TArray<TSharedPtr<FJsonValue>> InventoryArray;
    for (const FString& Tag : GameState.Player.InventoryTags)
    {
        InventoryArray.Add(MakeShareable(new FJsonValueString(Tag)));
    }
    PlayerJson->SetArrayField(TEXT("inventory_tags"), InventoryArray);

    JsonObject->SetObjectField(TEXT("player"), PlayerJson);

    // World
    TSharedPtr<FJsonObject> WorldJson = MakeShareable(new FJsonObject);
    WorldJson->SetStringField(TEXT("time_of_day"), GameState.World.TimeOfDay);
    if (!GameState.World.Weather.IsEmpty())
    {
        WorldJson->SetStringField(TEXT("weather"), GameState.World.Weather);
    }
    if (!GameState.World.TensionLevel.IsEmpty())
    {
        WorldJson->SetStringField(TEXT("tension_level"), GameState.World.TensionLevel);
    }
    JsonObject->SetObjectField(TEXT("world"), WorldJson);

    // Tags
    TArray<TSharedPtr<FJsonValue>> TagsArray;
    for (const FString& Tag : GameState.Tags)
    {
        TagsArray.Add(MakeShareable(new FJsonValueString(Tag)));
    }
    JsonObject->SetArrayField(TEXT("tags"), TagsArray);

    return JsonObject;
}

TSharedPtr<FJsonObject> UBigPlayClient::BuildPlayerIntentJson(const FBigPlayPlayerIntent& Intent)
{
    TSharedPtr<FJsonObject> JsonObject = MakeShareable(new FJsonObject);

    // Type
    FString TypeString;
    switch (Intent.Type)
    {
        case EBigPlayIntentType::TalkToNPC:
            TypeString = TEXT("talk_to_npc");
            break;
        case EBigPlayIntentType::EnterScene:
            TypeString = TEXT("enter_scene");
            break;
        case EBigPlayIntentType::RequestHint:
            TypeString = TEXT("request_hint");
            break;
        case EBigPlayIntentType::DebugSummary:
            TypeString = TEXT("debug_summary");
            break;
    }
    JsonObject->SetStringField(TEXT("type"), TypeString);

    if (!Intent.TargetNPCId.IsEmpty())
    {
        JsonObject->SetStringField(TEXT("target_npc_id"), Intent.TargetNPCId);
    }

    if (!Intent.RawInput.IsEmpty())
    {
        JsonObject->SetStringField(TEXT("raw_input"), Intent.RawInput);
    }

    return JsonObject;
}

TSharedPtr<FJsonObject> UBigPlayClient::BuildNPCJson(const FBigPlayNPC& NPC)
{
    TSharedPtr<FJsonObject> JsonObject = MakeShareable(new FJsonObject);

    JsonObject->SetStringField(TEXT("id"), NPC.Id);
    JsonObject->SetStringField(TEXT("name"), NPC.Name);
    JsonObject->SetStringField(TEXT("role"), NPC.Role);
    JsonObject->SetStringField(TEXT("mood"), NPC.Mood);
    JsonObject->SetStringField(TEXT("location"), NPC.Location);

    // Flags
    TSharedPtr<FJsonObject> FlagsJson = MakeShareable(new FJsonObject);
    for (const auto& Pair : NPC.Flags)
    {
        FlagsJson->SetBoolField(Pair.Key, Pair.Value);
    }
    JsonObject->SetObjectField(TEXT("flags"), FlagsJson);

    return JsonObject;
}

FBigPlayAction UBigPlayClient::ParseActionResponse(TSharedPtr<FJsonObject> JsonObject)
{
    FBigPlayAction Action;

    // Mode
    FString ModeString = JsonObject->GetStringField(TEXT("mode"));
    if (ModeString == TEXT("npc_dialogue"))
        Action.Mode = EBigPlayActionMode::NPCDialogue;
    else if (ModeString == TEXT("hint"))
        Action.Mode = EBigPlayActionMode::Hint;
    else if (ModeString == TEXT("world_reaction"))
        Action.Mode = EBigPlayActionMode::WorldReaction;
    else if (ModeString == TEXT("dev_debug"))
        Action.Mode = EBigPlayActionMode::DevDebug;

    // Priority
    FString PriorityString = JsonObject->GetStringField(TEXT("priority"));
    if (PriorityString == TEXT("low"))
        Action.Priority = EBigPlayPriority::Low;
    else if (PriorityString == TEXT("medium"))
        Action.Priority = EBigPlayPriority::Medium;
    else if (PriorityString == TEXT("high"))
        Action.Priority = EBigPlayPriority::High;

    // Dialogue
    if (JsonObject->HasField(TEXT("dialogue")))
    {
        const TArray<TSharedPtr<FJsonValue>>* DialogueArray;
        if (JsonObject->TryGetArrayField(TEXT("dialogue"), DialogueArray))
        {
            for (const TSharedPtr<FJsonValue>& Value : *DialogueArray)
            {
                TSharedPtr<FJsonObject> DialogueObj = Value->AsObject();
                if (DialogueObj.IsValid())
                {
                    FBigPlayDialogueLine Line;
                    Line.NPCId = DialogueObj->GetStringField(TEXT("npc_id"));
                    Line.Text = DialogueObj->GetStringField(TEXT("text"));
                    if (DialogueObj->HasField(TEXT("tone")))
                    {
                        Line.Tone = DialogueObj->GetStringField(TEXT("tone"));
                    }
                    Action.Dialogue.Add(Line);
                }
            }
        }
    }

    // Hint text
    if (JsonObject->HasField(TEXT("hint_text")))
    {
        Action.HintText = JsonObject->GetStringField(TEXT("hint_text"));
    }

    // World reaction
    if (JsonObject->HasField(TEXT("world_reaction")))
    {
        TSharedPtr<FJsonObject> WorldReactionObj = JsonObject->GetObjectField(TEXT("world_reaction"));
        if (WorldReactionObj.IsValid())
        {
            Action.WorldReaction.Description = WorldReactionObj->GetStringField(TEXT("description"));

            if (WorldReactionObj->HasField(TEXT("flag_changes")))
            {
                TSharedPtr<FJsonObject> FlagChangesObj = WorldReactionObj->GetObjectField(TEXT("flag_changes"));
                for (const auto& Pair : FlagChangesObj->Values)
                {
                    Action.WorldReaction.FlagChanges.Add(Pair.Key, Pair.Value->AsBool());
                }
            }
        }
    }

    // Debug notes
    if (JsonObject->HasField(TEXT("debug_notes")))
    {
        Action.DebugNotes = JsonObject->GetStringField(TEXT("debug_notes"));
    }

    return Action;
}
