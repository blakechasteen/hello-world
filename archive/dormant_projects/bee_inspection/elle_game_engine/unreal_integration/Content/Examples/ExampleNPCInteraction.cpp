// ExampleNPCInteraction.cpp
// Example: How to use BigPlay Engine in Unreal C++
// Version: 1.0.0

#include "ExampleNPCInteraction.h"
#include "BigPlayClient.h"
#include "BigPlayBlueprintLibrary.h"
#include "Kismet/GameplayStatics.h"

AExampleNPCInteraction::AExampleNPCInteraction()
{
	PrimaryActorTick.bCanEverTick = true;
}

void AExampleNPCInteraction::BeginPlay()
{
	Super::BeginPlay();

	// Initialize BigPlay client
	BigPlayClient = NewObject<UBigPlayClient>(this);
	BigPlayClient->Initialize(TEXT("http://localhost:8000"));

	UE_LOG(LogTemp, Log, TEXT("[Example] BigPlay client initialized"));
}

void AExampleNPCInteraction::Tick(float DeltaTime)
{
	Super::Tick(DeltaTime);
}

void AExampleNPCInteraction::TalkToNPC(const FString& NPCId, const FString& PlayerMessage)
{
	if (!BigPlayClient)
	{
		UE_LOG(LogTemp, Error, TEXT("[Example] BigPlay client not initialized"));
		return;
	}

	UE_LOG(LogTemp, Log, TEXT("[Example] Player says: \"%s\" to %s"), *PlayerMessage, *NPCId);

	// Create callbacks
	UBigPlayClient::FOnActionReceived OnSuccess;
	OnSuccess.BindUObject(this, &AExampleNPCInteraction::OnNPCDialogueReceived);

	UBigPlayClient::FOnRequestFailed OnFailure;
	OnFailure.BindUObject(this, &AExampleNPCInteraction::OnRequestFailed);

	// Send request
	BigPlayClient->GetNPCDialogue(
		NPCId,
		CurrentSceneId,
		PlayerMessage,
		OnSuccess,
		OnFailure
	);
}

void AExampleNPCInteraction::OnNPCDialogueReceived(const FBigPlayAction& Action)
{
	if (!Action.HasDialogue())
	{
		UE_LOG(LogTemp, Warning, TEXT("[Example] No dialogue in action"));
		return;
	}

	// Get first dialogue line
	const FBigPlayDialogueLine& Line = Action.Dialogue[0];

	UE_LOG(LogTemp, Log, TEXT("[Example] %s says: \"%s\" %s"),
		*Line.NPCId,
		*Line.Text,
		*Line.GetToneEmoji()
	);

	// Display in UI (implement your own UI logic here)
	DisplayDialogueInUI(Line.NPCId, Line.Text, Line.Tone);

	// Update NPC mood based on tone
	UpdateNPCMood(Line.NPCId, Line.Tone);
}

void AExampleNPCInteraction::OnRequestFailed(const FString& ErrorMessage)
{
	UE_LOG(LogTemp, Error, TEXT("[Example] Request failed: %s"), *ErrorMessage);

	// Show error to player (implement your own error UI)
	DisplayErrorInUI(ErrorMessage);
}

void AExampleNPCInteraction::DisplayDialogueInUI(
	const FString& NPCId,
	const FString& Text,
	const FString& Tone)
{
	// TODO: Implement your UI logic here
	// Examples:
	// - Show dialogue box with NPC name and text
	// - Play voice audio if available
	// - Animate NPC facial expression based on tone
	// - Add to dialogue history

	UE_LOG(LogTemp, Log, TEXT("[Example] TODO: Display dialogue in UI"));
}

void AExampleNPCInteraction::UpdateNPCMood(const FString& NPCId, const FString& Tone)
{
	// TODO: Implement NPC mood updates
	// Examples:
	// - Update NPC animation state
	// - Change NPC facial expression
	// - Adjust NPC behavior flags

	UE_LOG(LogTemp, Log, TEXT("[Example] TODO: Update NPC mood to %s"), *Tone);
}

void AExampleNPCInteraction::DisplayErrorInUI(const FString& ErrorMessage)
{
	// TODO: Implement error UI
	// Examples:
	// - Show toast notification
	// - Display in console log widget
	// - Retry mechanism

	UE_LOG(LogTemp, Error, TEXT("[Example] TODO: Display error UI"));
}

// Advanced example: Complete game state with multiple NPCs
void AExampleNPCInteraction::AdvancedExample()
{
	// Build complete game state
	FBigPlayGameState GameState;
	GameState.SceneId = TEXT("village_tavern");

	// Add multiple NPCs
	FBigPlayNPC Innkeeper;
	Innkeeper.Id = TEXT("innkeeper");
	Innkeeper.Name = TEXT("Bob");
	Innkeeper.Role = TEXT("innkeeper");
	Innkeeper.Mood = TEXT("cheerful");
	Innkeeper.Location = TEXT("behind_counter");
	Innkeeper.Flags.Add(TEXT("quest_available"), true);
	GameState.NPCs.Add(Innkeeper);

	FBigPlayNPC Guard;
	Guard.Id = TEXT("guard");
	Guard.Name = TEXT("Captain Marcus");
	Guard.Role = TEXT("guard_captain");
	Guard.Mood = TEXT("serious");
	Guard.Location = TEXT("entrance");
	Guard.Flags.Add(TEXT("suspicious_of_player"), false);
	GameState.NPCs.Add(Guard);

	// Set player state
	GameState.Player.Name = TEXT("Hero");
	GameState.Player.Location = TEXT("tavern_main_hall");
	GameState.Player.QuestStage = TEXT("seeking_information");
	GameState.Player.Reputation = TEXT("stranger");
	GameState.Player.Traits.Add(TEXT("charisma"), 7);
	GameState.Player.Traits.Add(TEXT("strength"), 5);
	GameState.Player.InventoryTags.Add(TEXT("gold_pouch"));
	GameState.Player.InventoryTags.Add(TEXT("travelers_cloak"));

	// Set world state
	GameState.World.TimeOfDay = TEXT("evening");
	GameState.World.Weather = TEXT("rainy");
	GameState.World.TensionLevel = TEXT("calm");

	// Create player intent
	FBigPlayPlayerIntent Intent;
	Intent.Type = EBigPlayIntentType::TalkToNPC;
	Intent.TargetNPCId = TEXT("innkeeper");
	Intent.RawInput = TEXT("I'm looking for information about the missing caravan.");

	// Send complete action
	UBigPlayClient::FOnActionReceived OnSuccess;
	OnSuccess.BindUObject(this, &AExampleNPCInteraction::OnNPCDialogueReceived);

	UBigPlayClient::FOnRequestFailed OnFailure;
	OnFailure.BindUObject(this, &AExampleNPCInteraction::OnRequestFailed);

	BigPlayClient->SendGameAction(GameState, Intent, OnSuccess, OnFailure);
}
