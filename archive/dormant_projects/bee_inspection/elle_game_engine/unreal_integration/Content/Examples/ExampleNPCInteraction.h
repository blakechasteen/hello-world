// ExampleNPCInteraction.h
// Example: How to use BigPlay Engine in Unreal C++
// Version: 1.0.0

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "BigPlayModels.h"
#include "ExampleNPCInteraction.generated.h"

UCLASS()
class BIGPLAYENGINE_API AExampleNPCInteraction : public AActor
{
	GENERATED_BODY()

public:
	AExampleNPCInteraction();

protected:
	virtual void BeginPlay() override;

public:
	virtual void Tick(float DeltaTime) override;

	/**
	 * Talk to an NPC
	 * @param NPCId - NPC identifier (e.g., "innkeeper")
	 * @param PlayerMessage - What the player said
	 */
	UFUNCTION(BlueprintCallable, Category = "BigPlay|Example")
	void TalkToNPC(const FString& NPCId, const FString& PlayerMessage);

	/**
	 * Advanced example showing complete game state
	 */
	UFUNCTION(BlueprintCallable, Category = "BigPlay|Example")
	void AdvancedExample();

private:
	// Callbacks
	void OnNPCDialogueReceived(const FBigPlayAction& Action);
	void OnRequestFailed(const FString& ErrorMessage);

	// UI helpers
	void DisplayDialogueInUI(const FString& NPCId, const FString& Text, const FString& Tone);
	void UpdateNPCMood(const FString& NPCId, const FString& Tone);
	void DisplayErrorInUI(const FString& ErrorMessage);

	// BigPlay client instance
	UPROPERTY()
	class UBigPlayClient* BigPlayClient;

	// Current scene ID
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "BigPlay", meta = (AllowPrivateAccess = "true"))
	FString CurrentSceneId = TEXT("village_tavern");
};
