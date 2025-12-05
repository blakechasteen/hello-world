// BigPlayClient.h
// BigPlay Engine - Unreal Engine Integration
// Version: 1.0.0
// Copyright (c) 2025 BigPlay

#pragma once

#include "CoreMinimal.h"
#include "Http.h"
#include "Json.h"
#include "BigPlayModels.h"
#include "BigPlayClient.generated.h"

/**
 * HTTP client for BigPlay Engine API
 *
 * Usage:
 *   UBigPlayClient* Client = NewObject<UBigPlayClient>();
 *   Client->Initialize("http://localhost:8000");
 *   Client->GetNPCDialogue("innkeeper", "village_tavern", "Hello!", FOnActionReceived::CreateLambda([](const FBigPlayAction& Action) {
 *       UE_LOG(LogTemp, Log, TEXT("NPC: %s"), *Action.GetDialogueText());
 *   }));
 */
UCLASS(BlueprintType)
class BIGPLAYENGINE_API UBigPlayClient : public UObject
{
    GENERATED_BODY()

public:
    // Delegates
    DECLARE_DYNAMIC_DELEGATE_OneParam(FOnActionReceived, const FBigPlayAction&, Action);
    DECLARE_DYNAMIC_DELEGATE_OneParam(FOnRequestFailed, const FString&, ErrorMessage);

    /**
     * Initialize the client with BigPlay API URL
     * @param InBaseURL - BigPlay server URL (e.g., "http://localhost:8000")
     */
    UFUNCTION(BlueprintCallable, Category = "BigPlay")
    void Initialize(const FString& InBaseURL);

    /**
     * Get NPC dialogue for player message
     * @param NPCId - Unique NPC identifier (e.g., "innkeeper")
     * @param SceneId - Current scene/location (e.g., "village_tavern")
     * @param PlayerMessage - What the player said
     * @param OnSuccess - Callback when action received
     * @param OnFailure - Callback on error
     */
    UFUNCTION(BlueprintCallable, Category = "BigPlay|Actions")
    void GetNPCDialogue(
        const FString& NPCId,
        const FString& SceneId,
        const FString& PlayerMessage,
        FOnActionReceived OnSuccess,
        FOnRequestFailed OnFailure
    );

    /**
     * Get hint for stuck player
     * @param SceneId - Current scene/location
     * @param QuestStage - Optional quest stage
     * @param OnSuccess - Callback when action received
     * @param OnFailure - Callback on error
     */
    UFUNCTION(BlueprintCallable, Category = "BigPlay|Actions")
    void GetHint(
        const FString& SceneId,
        const FString& QuestStage,
        FOnActionReceived OnSuccess,
        FOnRequestFailed OnFailure
    );

    /**
     * Get world reaction when entering scene
     * @param SceneId - Scene being entered
     * @param TimeOfDay - Current time (e.g., "morning", "afternoon")
     * @param Weather - Current weather (optional)
     * @param OnSuccess - Callback when action received
     * @param OnFailure - Callback on error
     */
    UFUNCTION(BlueprintCallable, Category = "BigPlay|Actions")
    void GetWorldReaction(
        const FString& SceneId,
        const FString& TimeOfDay,
        const FString& Weather,
        FOnActionReceived OnSuccess,
        FOnRequestFailed OnFailure
    );

    /**
     * Send complete game action request (advanced)
     * @param GameState - Complete game state snapshot
     * @param PlayerIntent - Player intent details
     * @param OnSuccess - Callback when action received
     * @param OnFailure - Callback on error
     */
    UFUNCTION(BlueprintCallable, Category = "BigPlay|Advanced")
    void SendGameAction(
        const FBigPlayGameState& GameState,
        const FBigPlayPlayerIntent& PlayerIntent,
        FOnActionReceived OnSuccess,
        FOnRequestFailed OnFailure
    );

    /**
     * Check if BigPlay server is healthy
     * @param OnSuccess - Callback on success
     * @param OnFailure - Callback on error
     */
    UFUNCTION(BlueprintCallable, Category = "BigPlay|System")
    void CheckHealth(
        FOnActionReceived OnSuccess,
        FOnRequestFailed OnFailure
    );

    /**
     * Get current API base URL
     */
    UFUNCTION(BlueprintPure, Category = "BigPlay|System")
    FString GetBaseURL() const { return BaseURL; }

    /**
     * Set request timeout in seconds (default: 30)
     */
    UFUNCTION(BlueprintCallable, Category = "BigPlay|System")
    void SetTimeout(float TimeoutSeconds) { Timeout = TimeoutSeconds; }

private:
    // HTTP helpers
    void SendRequest(
        const FString& Endpoint,
        const FString& Method,
        TSharedPtr<FJsonObject> JsonPayload,
        FOnActionReceived OnSuccess,
        FOnRequestFailed OnFailure
    );

    void OnResponseReceived(
        FHttpRequestPtr Request,
        FHttpResponsePtr Response,
        bool bWasSuccessful,
        FOnActionReceived OnSuccess,
        FOnRequestFailed OnFailure
    );

    // Build JSON payloads
    TSharedPtr<FJsonObject> BuildGameStateJson(const FBigPlayGameState& GameState);
    TSharedPtr<FJsonObject> BuildPlayerIntentJson(const FBigPlayPlayerIntent& Intent);
    TSharedPtr<FJsonObject> BuildNPCJson(const FBigPlayNPC& NPC);

    // Parse JSON response
    FBigPlayAction ParseActionResponse(TSharedPtr<FJsonObject> JsonObject);

    // Configuration
    FString BaseURL;
    float Timeout = 30.0f;
};
