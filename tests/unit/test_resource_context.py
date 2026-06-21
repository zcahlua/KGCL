from kgcl.resources.functional_groups import functional_group_resource_context, get_functional_group_resource_config


def test_resource_context_restores_sequential_and_nested(tmp_path):
    original = get_functional_group_resource_config()
    one = tmp_path / 'one'
    two = tmp_path / 'two'
    with functional_group_resource_context(resource_root=one, root_dir=one):
        assert get_functional_group_resource_config().resource_root == one.resolve()
        with functional_group_resource_context(resource_root=two, root_dir=two):
            assert get_functional_group_resource_config().resource_root == two.resolve()
        assert get_functional_group_resource_config().resource_root == one.resolve()
    assert get_functional_group_resource_config() == original
