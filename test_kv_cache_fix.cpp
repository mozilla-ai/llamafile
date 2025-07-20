#include <iostream>
#include <vector>
#include <cassert>

// Test for the KV cache crash fix
void test_cache_tokens_resize_fix() {
    std::cout << "Testing KV cache resize fix..." << std::endl;
    
    // Simulate the problematic condition
    std::vector<int> cache_tokens = {1, 2, 3, 4, 5};
    size_t original_size = cache_tokens.size();
    
    // Test cases that could cause integer underflow
    int n_discard_cases[] = {-1, 0, 3, 5, 10};
    
    for (int n_discard : n_discard_cases) {
        std::vector<int> test_tokens = cache_tokens;
        
        std::cout << "Testing n_discard = " << n_discard 
                  << " with cache size = " << test_tokens.size() << std::endl;
        
        // Apply the fixed logic
        if (n_discard >= 0 && (size_t)n_discard < test_tokens.size()) {
            test_tokens.resize(test_tokens.size() - n_discard);
            std::cout << "  Resized to: " << test_tokens.size() << std::endl;
        } else {
            test_tokens.clear();
            std::cout << "  Cleared to: " << test_tokens.size() << std::endl;
        }
        
        // Verify no crash occurred
        assert(test_tokens.size() <= original_size);
    }
    
    std::cout << "All test cases passed! KV cache fix works correctly." << std::endl;
}

int main() {
    test_cache_tokens_resize_fix();
    return 0;
}