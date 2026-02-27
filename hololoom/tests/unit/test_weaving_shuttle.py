"""
Unit tests for weaving_shuttle.py compatibility shim.

This module tests the deprecated WeavingShuttle compatibility layer.
Since WeavingShuttle is just an alias to WeavingOrchestrator, we test:
- Deprecation warning is emitted
- Alias works correctly
- Imported components match canonical versions
- Backward compatibility is maintained
"""

import pytest
import warnings
import importlib


class TestDeprecationWarning:
    """Test that deprecation warning is properly emitted."""

    def test_import_emits_deprecation_warning(self):
        """Importing weaving_shuttle should emit DeprecationWarning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            # Import and reload to force warning emission
            import hololoom.weaving_shuttle
            importlib.reload(HoloLoom.weaving_shuttle)

            # Should have at least one warning
            assert len(w) >= 1, "No warnings emitted"

            # Should be DeprecationWarning
            assert issubclass(w[-1].category, DeprecationWarning), \
                f"Warning is {w[-1].category}, not DeprecationWarning"

            # Warning message should mention WeavingOrchestrator
            warning_message = str(w[-1].message)
            assert "WeavingOrchestrator" in warning_message, \
                "Warning doesn't mention WeavingOrchestrator"
            assert "deprecated" in warning_message.lower(), \
                "Warning doesn't mention deprecation"

    def test_deprecation_warning_content(self):
        """Deprecation warning should provide migration instructions."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            import hololoom.weaving_shuttle
            importlib.reload(HoloLoom.weaving_shuttle)

            # Ensure warning was captured
            assert len(w) >= 1, "No warnings emitted"

            warning_message = str(w[-1].message)

            # Should mention removal in 2.0
            assert "2.0" in warning_message, \
                "Warning doesn't mention version 2.0 removal"

            # Should mention import path
            assert "hololoom.weaving_orchestrator" in warning_message, \
                "Warning doesn't provide correct import path"


class TestCompatibilityAlias:
    """Test that WeavingShuttle alias works correctly."""

    def test_weaving_shuttle_is_alias(self):
        """WeavingShuttle should be an alias to WeavingOrchestrator."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            from hololoom.weaving_shuttle import WeavingShuttle
            from hololoom.weaving_orchestrator import WeavingOrchestrator

            # Should be the same class
            assert WeavingShuttle is WeavingOrchestrator, \
                "WeavingShuttle is not an alias to WeavingOrchestrator"

    def test_weaving_orchestrator_available(self):
        """weaving_shuttle should also export WeavingOrchestrator."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            from hololoom.weaving_shuttle import WeavingOrchestrator

            # Should be importable
            assert WeavingOrchestrator is not None

    def test_tool_executor_available(self):
        """ToolExecutor should be available through compatibility shim."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            from hololoom.weaving_shuttle import ToolExecutor

            # Should be importable
            assert ToolExecutor is not None

    def test_yarn_graph_available(self):
        """YarnGraph should be available through compatibility shim."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            from hololoom.weaving_shuttle import YarnGraph

            # Should be importable
            assert YarnGraph is not None


class TestExports:
    """Test __all__ exports."""

    def test_all_exports_defined(self):
        """__all__ should be defined with correct exports."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            import hololoom.weaving_shuttle as shuttle_module

            # Should have __all__
            assert hasattr(shuttle_module, '__all__'), \
                "Module doesn't define __all__"

            # Should export expected items
            expected_exports = {'WeavingShuttle', 'WeavingOrchestrator', 'ToolExecutor', 'YarnGraph'}
            actual_exports = set(shuttle_module.__all__)

            assert expected_exports == actual_exports, \
                f"Expected {expected_exports}, got {actual_exports}"

    def test_all_exports_importable(self):
        """All items in __all__ should be importable."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            import hololoom.weaving_shuttle as shuttle_module

            for name in shuttle_module.__all__:
                assert hasattr(shuttle_module, name), \
                    f"{name} in __all__ but not defined in module"

                obj = getattr(shuttle_module, name)
                assert obj is not None, \
                    f"{name} is None"


class TestBackwardCompatibility:
    """Test that old code using WeavingShuttle still works."""

    @pytest.mark.integration
    def test_can_instantiate_via_alias(self, bare_config, test_shards):
        """Old code using WeavingShuttle should still work.

        NOTE: This is an integration test that creates a real WeavingOrchestrator.
        Skip in unit test runs with: pytest -m "not integration"
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            from hololoom.weaving_shuttle import WeavingShuttle

            # Should be able to create instance
            shuttle = WeavingShuttle(cfg=bare_config, shards=test_shards)

            # Should have expected attributes
            assert hasattr(shuttle, 'weave'), \
                "WeavingShuttle doesn't have weave method"
            assert hasattr(shuttle, 'close'), \
                "WeavingShuttle doesn't have close method"

    @pytest.mark.integration
    def test_isinstance_works_both_ways(self, bare_config, test_shards):
        """isinstance should work with both WeavingShuttle and WeavingOrchestrator.

        NOTE: This is an integration test that creates a real WeavingOrchestrator.
        Skip in unit test runs with: pytest -m "not integration"
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            from hololoom.weaving_shuttle import WeavingShuttle
            from hololoom.weaving_orchestrator import WeavingOrchestrator

            instance = WeavingShuttle(cfg=bare_config, shards=test_shards)

            # Should pass isinstance for both
            assert isinstance(instance, WeavingShuttle), \
                "Instance not recognized as WeavingShuttle"
            assert isinstance(instance, WeavingOrchestrator), \
                "Instance not recognized as WeavingOrchestrator"

    def test_type_equality(self):
        """type(WeavingShuttle) should equal type(WeavingOrchestrator)."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            from hololoom.weaving_shuttle import WeavingShuttle
            from hololoom.weaving_orchestrator import WeavingOrchestrator

            # Types should be identical
            assert type(WeavingShuttle) == type(WeavingOrchestrator), \
                "WeavingShuttle and WeavingOrchestrator have different types"


class TestDocumentation:
    """Test module documentation."""

    def test_module_has_docstring(self):
        """Module should have comprehensive docstring."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            import hololoom.weaving_shuttle as shuttle_module

            # Should have docstring
            assert shuttle_module.__doc__ is not None, \
                "Module has no docstring"
            assert len(shuttle_module.__doc__) > 100, \
                "Module docstring is too short"

    def test_docstring_mentions_deprecation(self):
        """Docstring should clearly state deprecation."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            import hololoom.weaving_shuttle as shuttle_module

            docstring = shuttle_module.__doc__.lower()

            # Should mention deprecation
            assert "deprecated" in docstring, \
                "Docstring doesn't mention deprecation"

            # Should mention compatibility
            assert "compatibility" in docstring or "backward" in docstring, \
                "Docstring doesn't mention backward compatibility"

    def test_docstring_provides_migration_path(self):
        """Docstring should tell users how to migrate."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            import hololoom.weaving_shuttle as shuttle_module

            docstring = shuttle_module.__doc__

            # Should mention new import path
            assert "WeavingOrchestrator" in docstring, \
                "Docstring doesn't mention WeavingOrchestrator"
            assert "weaving_orchestrator" in docstring, \
                "Docstring doesn't provide import path"


class TestFutureRemoval:
    """Test documentation about future removal."""

    def test_removal_version_documented(self):
        """Documentation should specify when shim will be removed."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            import hololoom.weaving_shuttle as shuttle_module

            docstring = shuttle_module.__doc__

            # Should mention version 2.0
            assert "2.0" in docstring, \
                "Docstring doesn't specify removal version"

    def test_warning_mentions_removal_version(self):
        """Deprecation warning should specify removal version."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            import hololoom.weaving_shuttle
            importlib.reload(HoloLoom.weaving_shuttle)

            # Ensure warning was captured
            assert len(w) >= 1, "No warnings emitted"

            warning_message = str(w[-1].message)

            # Should mention 2.0
            assert "2.0" in warning_message, \
                "Warning doesn't specify removal version"
