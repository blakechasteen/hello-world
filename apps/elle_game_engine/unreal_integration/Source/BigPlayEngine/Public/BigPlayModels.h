// BigPlayModels.h
// BigPlay Engine - Data Models for Unreal
// Version: 1.0.0

#pragma once

#include "CoreMinimal.h"
#include "BigPlayModels.generated.h"

/**
 * NPC state snapshot
 */
USTRUCT(BlueprintType)
struct FBigPlayNPC
{
    GENERATED_BODY()

    UPROPERTY(BlueprintReadWrite, Category = "BigPlay")
    FString Id;

    UPROPERTY(BlueprintReadWrite, Category = "BigPlay")
    FString Name;

    UPROPERTY(BlueprintReadWrite, Category = "BigPlay")
    FString Role;

    UPROPERTY(BlueprintReadWrite, Category = "BigPlay")
    FString Mood;

    UPROPERTY(BlueprintReadWrite, Category = "BigPlay")
    FString Location;

    UPROPERTY(BlueprintReadWrite, Category = "BigPlay")
    TMap<FString, bool> Flags;

    FBigPlayNPC()
        : Id(TEXT(""))
        , Name(TEXT(""))
        , Role(TEXT(""))
        , Mood(TEXT("neutral"))
        , Location(TEXT(""))
    {}
};

/**
 * Player state snapshot
 */
USTRUCT(BlueprintType)
struct FBigPlayPlayer
{
    GENERATED_BODY()

    UPROPERTY(BlueprintReadWrite, Category = "BigPlay")
    FString Name;

    UPROPERTY(BlueprintReadWrite, Category = "BigPlay")
    FString Location;

    UPROPERTY(BlueprintReadWrite, Category = "BigPlay")
    FString QuestStage;

    UPROPERTY(BlueprintReadWrite, Category = "BigPlay")
    FString Reputation;

    UPROPERTY(BlueprintReadWrite, Category = "BigPlay")
    TMap<FString, int32> Traits;

    UPROPERTY(BlueprintReadWrite, Category = "BigPlay")
    TArray<FString> InventoryTags;

    FBigPlayPlayer()
        : Name(TEXT("Player"))
        , Location(TEXT(""))
        , QuestStage(TEXT(""))
        , Reputation(TEXT("neutral"))
    {}
};

/**
 * World state snapshot
 */
USTRUCT(BlueprintType)
struct FBigPlayWorld
{
    GENERATED_BODY()

    UPROPERTY(BlueprintReadWrite, Category = "BigPlay")
    FString TimeOfDay;

    UPROPERTY(BlueprintReadWrite, Category = "BigPlay")
    FString Weather;

    UPROPERTY(BlueprintReadWrite, Category = "BigPlay")
    FString TensionLevel;

    FBigPlayWorld()
        : TimeOfDay(TEXT("afternoon"))
        , Weather(TEXT("clear"))
        , TensionLevel(TEXT("calm"))
    {}
};

/**
 * Complete game state
 */
USTRUCT(BlueprintType)
struct FBigPlayGameState
{
    GENERATED_BODY()

    UPROPERTY(BlueprintReadWrite, Category = "BigPlay")
    FString SceneId;

    UPROPERTY(BlueprintReadWrite, Category = "BigPlay")
    TArray<FBigPlayNPC> NPCs;

    UPROPERTY(BlueprintReadWrite, Category = "BigPlay")
    FBigPlayPlayer Player;

    UPROPERTY(BlueprintReadWrite, Category = "BigPlay")
    FBigPlayWorld World;

    UPROPERTY(BlueprintReadWrite, Category = "BigPlay")
    TArray<FString> Tags;

    FBigPlayGameState()
        : SceneId(TEXT(""))
    {}
};

/**
 * Player intent types
 */
UENUM(BlueprintType)
enum class EBigPlayIntentType : uint8
{
    TalkToNPC       UMETA(DisplayName = "Talk to NPC"),
    EnterScene      UMETA(DisplayName = "Enter Scene"),
    RequestHint     UMETA(DisplayName = "Request Hint"),
    DebugSummary    UMETA(DisplayName = "Debug Summary")
};

/**
 * Player intent
 */
USTRUCT(BlueprintType)
struct FBigPlayPlayerIntent
{
    GENERATED_BODY()

    UPROPERTY(BlueprintReadWrite, Category = "BigPlay")
    EBigPlayIntentType Type;

    UPROPERTY(BlueprintReadWrite, Category = "BigPlay")
    FString TargetNPCId;

    UPROPERTY(BlueprintReadWrite, Category = "BigPlay")
    FString RawInput;

    FBigPlayPlayerIntent()
        : Type(EBigPlayIntentType::TalkToNPC)
        , TargetNPCId(TEXT(""))
        , RawInput(TEXT(""))
    {}
};

/**
 * Action modes
 */
UENUM(BlueprintType)
enum class EBigPlayActionMode : uint8
{
    NPCDialogue     UMETA(DisplayName = "NPC Dialogue"),
    Hint            UMETA(DisplayName = "Hint"),
    WorldReaction   UMETA(DisplayName = "World Reaction"),
    DevDebug        UMETA(DisplayName = "Dev Debug")
};

/**
 * Priority levels
 */
UENUM(BlueprintType)
enum class EBigPlayPriority : uint8
{
    Low             UMETA(DisplayName = "Low"),
    Medium          UMETA(DisplayName = "Medium"),
    High            UMETA(DisplayName = "High")
};

/**
 * Dialogue line from NPC
 */
USTRUCT(BlueprintType)
struct FBigPlayDialogueLine
{
    GENERATED_BODY()

    UPROPERTY(BlueprintReadWrite, Category = "BigPlay")
    FString NPCId;

    UPROPERTY(BlueprintReadWrite, Category = "BigPlay")
    FString Text;

    UPROPERTY(BlueprintReadWrite, Category = "BigPlay")
    FString Tone;

    FBigPlayDialogueLine()
        : NPCId(TEXT(""))
        , Text(TEXT(""))
        , Tone(TEXT("neutral"))
    {}

    /**
     * Get emoji for tone (helper)
     */
    FString GetToneEmoji() const
    {
        if (Tone == TEXT("happy") || Tone == TEXT("warm")) return TEXT("😊");
        if (Tone == TEXT("angry") || Tone == TEXT("harsh")) return TEXT("😠");
        if (Tone == TEXT("sad") || Tone == TEXT("melancholy")) return TEXT("😢");
        if (Tone == TEXT("fearful") || Tone == TEXT("worried")) return TEXT("😰");
        if (Tone == TEXT("surprised")) return TEXT("😲");
        if (Tone == TEXT("confused")) return TEXT("😕");
        return TEXT("😐");
    }
};

/**
 * World reaction details
 */
USTRUCT(BlueprintType)
struct FBigPlayWorldReaction
{
    GENERATED_BODY()

    UPROPERTY(BlueprintReadWrite, Category = "BigPlay")
    FString Description;

    UPROPERTY(BlueprintReadWrite, Category = "BigPlay")
    TMap<FString, bool> FlagChanges;

    FBigPlayWorldReaction()
        : Description(TEXT(""))
    {}
};

/**
 * Complete action response from BigPlay
 */
USTRUCT(BlueprintType)
struct FBigPlayAction
{
    GENERATED_BODY()

    UPROPERTY(BlueprintReadWrite, Category = "BigPlay")
    EBigPlayActionMode Mode;

    UPROPERTY(BlueprintReadWrite, Category = "BigPlay")
    EBigPlayPriority Priority;

    UPROPERTY(BlueprintReadWrite, Category = "BigPlay")
    TArray<FBigPlayDialogueLine> Dialogue;

    UPROPERTY(BlueprintReadWrite, Category = "BigPlay")
    FString HintText;

    UPROPERTY(BlueprintReadWrite, Category = "BigPlay")
    FBigPlayWorldReaction WorldReaction;

    UPROPERTY(BlueprintReadWrite, Category = "BigPlay")
    FString DebugNotes;

    FBigPlayAction()
        : Mode(EBigPlayActionMode::NPCDialogue)
        , Priority(EBigPlayPriority::Medium)
        , HintText(TEXT(""))
        , DebugNotes(TEXT(""))
    {}

    /**
     * Helper: Check if action has dialogue
     */
    UFUNCTION(BlueprintPure, Category = "BigPlay")
    bool HasDialogue() const { return Dialogue.Num() > 0; }

    /**
     * Helper: Get first dialogue text (most common case)
     */
    UFUNCTION(BlueprintPure, Category = "BigPlay")
    FString GetDialogueText() const
    {
        return HasDialogue() ? Dialogue[0].Text : TEXT("");
    }

    /**
     * Helper: Get first NPC ID
     */
    UFUNCTION(BlueprintPure, Category = "BigPlay")
    FString GetNPCId() const
    {
        return HasDialogue() ? Dialogue[0].NPCId : TEXT("");
    }

    /**
     * Helper: Check if action has hint
     */
    UFUNCTION(BlueprintPure, Category = "BigPlay")
    bool HasHint() const { return !HintText.IsEmpty(); }

    /**
     * Helper: Check if action has world reaction
     */
    UFUNCTION(BlueprintPure, Category = "BigPlay")
    bool HasWorldReaction() const { return !WorldReaction.Description.IsEmpty(); }
};
