// BigPlayBlueprintLibrary.cpp
// Blueprint Function Library Implementation
// Version: 1.0.0

#include "BigPlayBlueprintLibrary.h"
#include "BigPlayClient.h"
#include "Engine/World.h"

// Static instance map
TMap<UWorld*, UBigPlayClient*> UBigPlayBlueprintLibrary::ClientInstances;

void UBigPlayBlueprintLibrary::QuickDialogue(
    UObject* WorldContextObject,
    const FString& NPCId,
    const FString& SceneId,
    const FString& PlayerMessage,
    const FScriptDelegate& OnSuccess,
    const FScriptDelegate& OnFailure)
{
    UBigPlayClient* Client = GetBigPlayClient(WorldContextObject);
    if (!Client)
    {
        UE_LOG(LogTemp, Error, TEXT("[BigPlay] Failed to get client instance"));
        return;
    }

    // Convert script delegates to native delegates
    UBigPlayClient::FOnActionReceived SuccessDelegate;
    SuccessDelegate.BindLambda([OnSuccess](const FBigPlayAction& Action) mutable
    {
        if (OnSuccess.IsBound())
        {
            // Call Blueprint callback
            // Note: Blueprint delegates need special handling
            UE_LOG(LogTemp, Log, TEXT("[BigPlay] Action received: %s"), *Action.GetDialogueText());
        }
    });

    UBigPlayClient::FOnRequestFailed FailureDelegate;
    FailureDelegate.BindLambda([OnFailure](const FString& ErrorMessage) mutable
    {
        if (OnFailure.IsBound())
        {
            UE_LOG(LogTemp, Error, TEXT("[BigPlay] Request failed: %s"), *ErrorMessage);
        }
    });

    Client->GetNPCDialogue(NPCId, SceneId, PlayerMessage, SuccessDelegate, FailureDelegate);
}

void UBigPlayBlueprintLibrary::QuickHint(
    UObject* WorldContextObject,
    const FString& SceneId,
    const FString& QuestStage,
    const FScriptDelegate& OnSuccess,
    const FScriptDelegate& OnFailure)
{
    UBigPlayClient* Client = GetBigPlayClient(WorldContextObject);
    if (!Client)
    {
        UE_LOG(LogTemp, Error, TEXT("[BigPlay] Failed to get client instance"));
        return;
    }

    UBigPlayClient::FOnActionReceived SuccessDelegate;
    SuccessDelegate.BindLambda([OnSuccess](const FBigPlayAction& Action) mutable
    {
        if (OnSuccess.IsBound())
        {
            UE_LOG(LogTemp, Log, TEXT("[BigPlay] Hint received: %s"), *Action.HintText);
        }
    });

    UBigPlayClient::FOnRequestFailed FailureDelegate;
    FailureDelegate.BindLambda([OnFailure](const FString& ErrorMessage) mutable
    {
        if (OnFailure.IsBound())
        {
            UE_LOG(LogTemp, Error, TEXT("[BigPlay] Request failed: %s"), *ErrorMessage);
        }
    });

    Client->GetHint(SceneId, QuestStage, SuccessDelegate, FailureDelegate);
}

FBigPlayNPC UBigPlayBlueprintLibrary::MakeNPC(
    const FString& Id,
    const FString& Name,
    const FString& Role,
    const FString& Mood,
    const FString& Location)
{
    FBigPlayNPC NPC;
    NPC.Id = Id;
    NPC.Name = Name;
    NPC.Role = Role;
    NPC.Mood = Mood.IsEmpty() ? TEXT("neutral") : Mood;
    NPC.Location = Location;
    return NPC;
}

FBigPlayPlayer UBigPlayBlueprintLibrary::MakePlayer(
    const FString& Name,
    const FString& Location,
    const FString& QuestStage,
    const FString& Reputation)
{
    FBigPlayPlayer Player;
    Player.Name = Name;
    Player.Location = Location;
    Player.QuestStage = QuestStage;
    Player.Reputation = Reputation.IsEmpty() ? TEXT("neutral") : Reputation;
    return Player;
}

FBigPlayWorld UBigPlayBlueprintLibrary::MakeWorld(
    const FString& TimeOfDay,
    const FString& Weather,
    const FString& TensionLevel)
{
    FBigPlayWorld World;
    World.TimeOfDay = TimeOfDay;
    World.Weather = Weather;
    World.TensionLevel = TensionLevel.IsEmpty() ? TEXT("calm") : TensionLevel;
    return World;
}

FBigPlayGameState UBigPlayBlueprintLibrary::MakeGameState(
    const FString& SceneId,
    const TArray<FBigPlayNPC>& NPCs,
    const FBigPlayPlayer& Player,
    const FBigPlayWorld& World)
{
    FBigPlayGameState GameState;
    GameState.SceneId = SceneId;
    GameState.NPCs = NPCs;
    GameState.Player = Player;
    GameState.World = World;
    return GameState;
}

FBigPlayPlayerIntent UBigPlayBlueprintLibrary::MakeIntentTalkToNPC(
    const FString& TargetNPCId,
    const FString& PlayerMessage)
{
    FBigPlayPlayerIntent Intent;
    Intent.Type = EBigPlayIntentType::TalkToNPC;
    Intent.TargetNPCId = TargetNPCId;
    Intent.RawInput = PlayerMessage;
    return Intent;
}

FBigPlayPlayerIntent UBigPlayBlueprintLibrary::MakeIntentEnterScene()
{
    FBigPlayPlayerIntent Intent;
    Intent.Type = EBigPlayIntentType::EnterScene;
    return Intent;
}

FBigPlayPlayerIntent UBigPlayBlueprintLibrary::MakeIntentRequestHint()
{
    FBigPlayPlayerIntent Intent;
    Intent.Type = EBigPlayIntentType::RequestHint;
    return Intent;
}

FString UBigPlayBlueprintLibrary::GetDialogueText(const FBigPlayAction& Action)
{
    return Action.GetDialogueText();
}

FString UBigPlayBlueprintLibrary::GetNPCId(const FBigPlayAction& Action)
{
    return Action.GetNPCId();
}

FString UBigPlayBlueprintLibrary::GetToneEmoji(const FBigPlayAction& Action)
{
    if (Action.HasDialogue())
    {
        return Action.Dialogue[0].GetToneEmoji();
    }
    return TEXT("😐");
}

bool UBigPlayBlueprintLibrary::IsHighPriority(const FBigPlayAction& Action)
{
    return Action.Priority == EBigPlayPriority::High;
}

UBigPlayClient* UBigPlayBlueprintLibrary::GetBigPlayClient(UObject* WorldContextObject)
{
    if (!WorldContextObject)
    {
        UE_LOG(LogTemp, Error, TEXT("[BigPlay] Invalid WorldContextObject"));
        return nullptr;
    }

    UWorld* World = WorldContextObject->GetWorld();
    if (!World)
    {
        UE_LOG(LogTemp, Error, TEXT("[BigPlay] Failed to get World from WorldContextObject"));
        return nullptr;
    }

    // Check if client already exists for this world
    UBigPlayClient** ExistingClient = ClientInstances.Find(World);
    if (ExistingClient && *ExistingClient)
    {
        return *ExistingClient;
    }

    // Create new client instance
    UBigPlayClient* NewClient = NewObject<UBigPlayClient>();
    NewClient->Initialize(TEXT("http://localhost:8000")); // Default URL

    // Cache for future use
    ClientInstances.Add(World, NewClient);

    UE_LOG(LogTemp, Log, TEXT("[BigPlay] Created new client instance for world"));

    return NewClient;
}
