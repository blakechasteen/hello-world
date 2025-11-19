## Elle Game Engine plugin for Godot.
##
## This plugin adds ElleClient as an autoload singleton and provides
## custom editor features for working with Elle's narrative system.
##
## Author: Elle Game Engine Team
## Date: 2025-11-16

@tool
extends EditorPlugin

const AUTOLOAD_NAME = "Elle"


func _enter_tree():
	# Add autoload singleton
	add_autoload_singleton(AUTOLOAD_NAME, "res://addons/elle_game_engine/ElleClient.gd")

	print("Elle Game Engine plugin enabled!")
	print("ElleClient is now available as autoload singleton: Elle")
	print("Configure Elle.base_url in Project Settings > Autoload > Elle")


func _exit_tree():
	# Remove autoload singleton
	remove_autoload_singleton(AUTOLOAD_NAME)

	print("Elle Game Engine plugin disabled!")
