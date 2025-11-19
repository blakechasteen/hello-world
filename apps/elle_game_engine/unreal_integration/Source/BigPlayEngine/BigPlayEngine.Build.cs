// BigPlayEngine.Build.cs
// Build Configuration for BigPlay Engine Plugin
// Version: 1.0.0

using UnrealBuildTool;

public class BigPlayEngine : ModuleRules
{
	public BigPlayEngine(ReadOnlyTargetRules Target) : base(Target)
	{
		PCHUsage = ModuleRules.PCHUsageMode.UseExplicitOrSharedPCHs;

		PublicIncludePaths.AddRange(
			new string[] {
				// Public include paths
			}
		);

		PrivateIncludePaths.AddRange(
			new string[] {
				// Private include paths
			}
		);

		PublicDependencyModuleNames.AddRange(
			new string[]
			{
				"Core",
				"CoreUObject",
				"Engine",
				"Http",
				"Json",
				"JsonUtilities"
			}
		);

		PrivateDependencyModuleNames.AddRange(
			new string[]
			{
				// Private dependencies
			}
		);

		DynamicallyLoadedModuleNames.AddRange(
			new string[]
			{
				// Dynamically loaded modules
			}
		);
	}
}
