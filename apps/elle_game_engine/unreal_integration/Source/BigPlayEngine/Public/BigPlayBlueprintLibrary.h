// BigPlayBlueprintLibrary.h
// Blueprint Function Library for BigPlay Engine
// Version: 1.0.0

#pragma once

#include "CoreMinimal.h"
#include "Kismet/BlueprintFunctionLibrary.h"
#include "BigPlayModels.h"
#include "BigPlayBlueprintLibrary.generated.h"

/**
 * Blueprint function library for BigPlay Engine
 *
 * Provides convenient Blueprint-callable functions for common operations.
 * Use these in Blueprint graphs for rapid prototyping.
 *
 * Example (Blueprint):
 *   UBigPlayBlueprintLibrary::QuickDialogue(
 *       "innkeeper",
 *       "village_tavern",
 *       "Hello!",
 *       [Callback Success],
 *       [Callback Failure]
 *   );
 */
UCLASS()
class BIGPLAYENGINE_API UBigPlayBlueprintLibrary : public UBlueprintFunctionLibrary
{
    GENERATED_BODY()

public:
    /**
     * Quick dialogue - simplified NPC conversation
     * @param NPCId - NPC identifier (e.g., "innkeeper")
     * @param SceneId - Current scene (e.g., "village_tavern")
     * @param PlayerMessage - What the player said
     * @param OnSuccess - Callback when dialogue received
     * @param OnFailure - Callback on error
     */
    UFUNCTION(BlueprintCallable, Category = "BigPlay|Quick Actions", meta = (WorldContext = "WorldContextObject"))
    static void QuickDialogue(
        UObject* WorldContextObject,
        const FString& NPCId,
        const FString& SceneId,
        const FString& PlayerMessage,
        const FScriptDelegate& OnSuccess,
        const FScriptDelegate& OnFailure
    );

    /**
     * Quick hint - get hint for stuck player
     * @param SceneId - Current scene
     * @param QuestStage - Optional quest stage
     * @param OnSuccess - Callback when hint received
     * @param OnFailure - Callback on error
     */
    UFUNCTION(BlueprintCallable, Category = "BigPlay|Quick Actions", meta = (WorldContext = "WorldContextObject"))
    static void QuickHint(
        UObject* WorldContextObject,
        const FString& SceneId,
        const FString& QuestStage,
        const FScriptDelegate& OnSuccess,
        const FScriptDelegate& OnFailure
    );

    /**
     * Create NPC state
     * @param Id - NPC identifier
     * @param Name - NPC display name
     * @param Role - NPC role (e.g., "merchant", "guard")
     * @param Mood - Current mood (optional, default: "neutral")
     * @param Location - Current location
     * @return FBigPlayNPC struct
     */
    UFUNCTION(BlueprintPure, Category = "BigPlay|Builders")
    static FBigPlayNPC MakeNPC(
        const FString& Id,
        const FString& Name,
        const FString& Role,
        const FString& Mood,
        const FString& Location
    );

    /**
     * Create player state
     * @param Name - Player name
     * @param Location - Current location
     * @param QuestStage - Optional quest stage
     * @param Reputation - Optional reputation
     * @return FBigPlayPlayer struct
     */
    UFUNCTION(BlueprintPure, Category = "BigPlay|Builders")
    static FBigPlayPlayer MakePlayer(
        const FString& Name,
        const FString& Location,
        const FString& QuestStage,
        const FString& Reputation
    );

    /**
     * Create world state
     * @param TimeOfDay - Time of day (e.g., "morning", "afternoon", "evening", "night")
     * @param Weather - Weather condition (optional)
     * @param TensionLevel - Tension level (optional, e.g., "calm", "tense", "critical")
     * @return FBigPlayWorld struct
     */
    UFUNCTION(BlueprintPure, Category = "BigPlay|Builders")
    static FBigPlayWorld MakeWorld(
        const FString& TimeOfDay,
        const FString& Weather,
        const FString& TensionLevel
    );

    /**
     * Create game state
     * @param SceneId - Scene identifier
     * @param NPCs - Array of NPCs in scene
     * @param Player - Player state
     * @param World - World state
     * @return FBigPlayGameState struct
     */
    UFUNCTION(BlueprintPure, Category = "BigPlay|Builders")
    static FBigPlayGameState MakeGameState(
        const FString& SceneId,
        const TArray<FBigPlayNPC>& NPCs,
        const FBigPlayPlayer& Player,
        const FBigPlayWorld& World
    );

    /**
     * Create player intent - talk to NPC
     * @param TargetNPCId - NPC to talk to
     * @param PlayerMessage - What the player said
     * @return FBigPlayPlayerIntent struct
     */
    UFUNCTION(BlueprintPure, Category = "BigPlay|Builders")
    static FBigPlayPlayerIntent MakeIntentTalkToNPC(
        const FString& TargetNPCId,
        const FString& PlayerMessage
    );

    /**
     * Create player intent - enter scene
     * @return FBigPlayPlayerIntent struct
     */
    UFUNCTION(BlueprintPure, Category = "BigPlay|Builders")
    static FBigPlayPlayerIntent MakeIntentEnterScene();

    /**
     * Create player intent - request hint
     * @return FBigPlayPlayerIntent struct
     */
    UFUNCTION(BlueprintPure, Category = "BigPlay|Builders")
    static FBigPlayPlayerIntent MakeIntentRequestHint();

    /**
     * Get dialogue text from action (first dialogue line)
     * @param Action - BigPlay action
     * @return Dialogue text (empty if no dialogue)
     */
    UFUNCTION(BlueprintPure, Category = "BigPlay|Helpers")
    static FString GetDialogueText(const FBigPlayAction& Action);

    /**
     * Get NPC ID from action (first dialogue line)
     * @param Action - BigPlay action
     * @return NPC ID (empty if no dialogue)
     */
    UFUNCTION(BlueprintPure, Category = "BigPlay|Helpers")
    static FString GetNPCId(const FBigPlayAction& Action);

    /**
     * Get tone emoji from action (first dialogue line)
     * @param Action - BigPlay action
     * @return Tone emoji (😐 if no dialogue)
     */
    UFUNCTION(BlueprintPure, Category = "BigPlay|Helpers")
    static FString GetToneEmoji(const FBigPlayAction& Action);

    /**
     * Check if action is high priority
     * @param Action - BigPlay action
     * @return True if high priority
     */
    UFUNCTION(BlueprintPure, Category = "BigPlay|Helpers")
    static bool IsHighPriority(const FBigPlayAction& Action);

    /**
     * Get BigPlay client (singleton-like access)
     * Creates client if it doesn't exist
     */
    UFUNCTION(BlueprintCallable, Category = "BigPlay|System", meta = (WorldContext = "WorldContextObject"))
    static class UBigPlayClient* GetBigPlayClient(UObject* WorldContextObject);

private:
    // Cached client instance (per world)
    static TMap<UWorld*, UBigPlayClient*> ClientInstances;
};
