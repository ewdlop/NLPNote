// Supernatural Christmas Tree Generator in Carbon Language
// Demonstrates Carbon's choice types (discriminated unions) with botanical taxonomy
// 
// Carbon is an experimental successor to C++ with modern type system features
// including choice types which are similar to Rust enums or discriminated unions
// in other languages.

package SupernaturalChristmasTree;

// Import Carbon standard library
import Carbon.Std;

// ============================================================================
// ENUMERATIONS
// ============================================================================

// Tree node discriminator for the choice type
choice TreeNodeType {
  Genus,
  Species,
  Ornament
}

// Ornament colors with Unicode emoji representation
choice OrnamentColor {
  Red,      // 🔴
  Gold,     // 🟡
  Blue,     // 🔵
  Green,    // 🟢
  Purple,   // 🟣
  White,    // ⚪
  Silver    // ⚫
}

// Magical effects for supernatural ornaments
choice MagicalEffect {
  Sparkle,   // ✨
  Star,      // ⭐
  Snowflake, // ❄️
  Glow,      // 💫
  Shine,     // 🌟
  Twinkle    // ✦
}

// ============================================================================
// DISCRIMINATED UNION NODE TYPES
// ============================================================================

// GenusNode represents the root of the tree (botanical genus)
class GenusNode {
  var name: String;
  var common_name: String;
  var family: String;
  var magical_power: f64;
  
  // Constructor
  fn Create(name: String, common_name: String, family: String, magical_power: f64) -> Self {
    return {
      .name = name,
      .common_name = common_name,
      .family = family,
      .magical_power = magical_power
    };
  }
  
  // Node type discriminator
  fn GetNodeType[self: Self]() -> TreeNodeType {
    return TreeNodeType.Genus;
  }
  
  fn Display[self: Self]() {
    Print("GenusNode: {0} ({1}) - Family: {2}, Power: {3}", 
          self.name, self.common_name, self.family, self.magical_power);
  }
}

// SpeciesNode represents branches (botanical species)
class SpeciesNode {
  var name: String;
  var genus: String;
  var height_level: i32;
  var supernatural_glow: f64;
  var characteristics: Vector(String);
  
  // Constructor
  fn Create(name: String, genus: String, height_level: i32, supernatural_glow: f64) -> Self {
    return {
      .name = name,
      .genus = genus,
      .height_level = height_level,
      .supernatural_glow = supernatural_glow,
      .characteristics = Vector(String).Create()
    };
  }
  
  // Node type discriminator
  fn GetNodeType[self: Self]() -> TreeNodeType {
    return TreeNodeType.Species;
  }
  
  fn Display[self: Self]() {
    Print("SpeciesNode: {0} - Level: {1}, Glow: {2}", 
          self.name, self.height_level, self.supernatural_glow);
  }
}

// OrnamentNode represents supernatural decorations
class OrnamentNode {
  var color: OrnamentColor;
  var effect: MagicalEffect;
  var intensity: f64;
  var semantic_meaning: String;
  var position_level: i32;
  var position_offset: i32;
  
  // Constructor
  fn Create(color: OrnamentColor, effect: MagicalEffect, intensity: f64,
            semantic_meaning: String, level: i32, offset: i32) -> Self {
    return {
      .color = color,
      .effect = effect,
      .intensity = intensity,
      .semantic_meaning = semantic_meaning,
      .position_level = level,
      .position_offset = offset
    };
  }
  
  // Node type discriminator
  fn GetNodeType[self: Self]() -> TreeNodeType {
    return TreeNodeType.Ornament;
  }
  
  fn Display[self: Self]() {
    Print("OrnamentNode: Color={0}, Effect={1}, Intensity={2}, Position=({3},{4})",
          self.color, self.effect, self.intensity, 
          self.position_level, self.position_offset);
  }
}

// ============================================================================
// DISCRIMINATED UNION (CHOICE TYPE)
// ============================================================================

// TreeNode is a choice type (discriminated union) that can be one of three variants
// This is Carbon's equivalent to Rust's enum with data or TypeScript's discriminated unions
choice TreeNode {
  Genus(GenusNode),
  Species(SpeciesNode),
  Ornament(OrnamentNode)
}

// Pattern matching helper for TreeNode choice type
fn GetTreeNodeType(node: TreeNode) -> TreeNodeType {
  // Carbon uses pattern matching with choice types
  match (node) {
    case TreeNode.Genus(genus_node) => {
      return TreeNodeType.Genus;
    }
    case TreeNode.Species(species_node) => {
      return TreeNodeType.Species;
    }
    case TreeNode.Ornament(ornament_node) => {
      return TreeNodeType.Ornament;
    }
  }
}

// Display function with pattern matching
fn DisplayNode(node: TreeNode) {
  match (node) {
    case TreeNode.Genus(genus_node) => {
      genus_node.Display();
    }
    case TreeNode.Species(species_node) => {
      species_node.Display();
    }
    case TreeNode.Ornament(ornament_node) => {
      ornament_node.Display();
    }
  }
}

// ============================================================================
// BOTANICAL DATABASE
// ============================================================================

// Structure to hold genus information
class GenusData {
  var common_name: String;
  var family: String;
  var species: Vector(String);
  var magical_power: f64;
  
  fn Create(common_name: String, family: String, magical_power: f64) -> Self {
    return {
      .common_name = common_name,
      .family = family,
      .species = Vector(String).Create(),
      .magical_power = magical_power
    };
  }
}

// Static botanical database class
class BotanicalDatabase {
  // Static database of Christmas tree genera
  fn GetAbiesData() -> GenusData {
    var data: GenusData = GenusData.Create("Fir", "Pinaceae", 0.9);
    data.species.Push("Abies alba");        // Silver Fir
    data.species.Push("Abies balsamea");    // Balsam Fir
    data.species.Push("Abies fraseri");     // Fraser Fir
    data.species.Push("Abies nordmanniana"); // Nordmann Fir
    data.species.Push("Abies procera");     // Noble Fir
    return data;
  }
  
  fn GetPiceaData() -> GenusData {
    var data: GenusData = GenusData.Create("Spruce", "Pinaceae", 0.85);
    data.species.Push("Picea abies");    // Norway Spruce
    data.species.Push("Picea pungens");  // Blue Spruce
    data.species.Push("Picea glauca");   // White Spruce
    data.species.Push("Picea omorika");  // Serbian Spruce
    return data;
  }
  
  fn GetPinusData() -> GenusData {
    var data: GenusData = GenusData.Create("Pine", "Pinaceae", 0.8);
    data.species.Push("Pinus sylvestris"); // Scots Pine
    data.species.Push("Pinus nigra");      // Austrian Pine
    data.species.Push("Pinus strobus");    // White Pine
    return data;
  }
  
  // Get genus data by name
  fn GetGenusData(genus_name: String) -> Optional(GenusData) {
    if (genus_name == "Abies") {
      return Optional(GenusData).Of(GetAbiesData());
    } else if (genus_name == "Picea") {
      return Optional(GenusData).Of(GetPiceaData());
    } else if (genus_name == "Pinus") {
      return Optional(GenusData).Of(GetPinusData());
    }
    return Optional(GenusData).None();
  }
  
  // List all available genera
  fn ListGenera() -> Vector(String) {
    var genera: Vector(String) = Vector(String).Create();
    genera.Push("Abies");
    genera.Push("Picea");
    genera.Push("Pinus");
    return genera;
  }
}

// ============================================================================
// SUPERNATURAL CHRISTMAS TREE
// ============================================================================

class SupernaturalChristmasTree {
  var genus_name: String;
  var root: Optional(GenusNode);
  var species_nodes: Vector(SpeciesNode);
  var ornaments: Vector(OrnamentNode);
  var all_nodes: Vector(TreeNode);
  var height: i32;
  var magical_aura: f64;
  
  // Constructor
  fn Create(genus_name: String) -> Self {
    var tree: Self = {
      .genus_name = genus_name,
      .root = Optional(GenusNode).None(),
      .species_nodes = Vector(SpeciesNode).Create(),
      .ornaments = Vector(OrnamentNode).Create(),
      .all_nodes = Vector(TreeNode).Create(),
      .height = 0,
      .magical_aura = 0.0
    };
    
    tree.Initialize();
    return tree;
  }
  
  // Initialize the tree from botanical database
  fn Initialize[addr self: Self*]() {
    var genus_data_opt: Optional(GenusData) = BotanicalDatabase.GetGenusData(self->genus_name);
    
    // If genus not found, default to Abies
    if (not genus_data_opt.HasValue()) {
      self->genus_name = "Abies";
      genus_data_opt = BotanicalDatabase.GetGenusData("Abies");
    }
    
    var genus_data: GenusData = genus_data_opt.Value();
    
    // Create root genus node
    var root_node: GenusNode = GenusNode.Create(
      self->genus_name,
      genus_data.common_name,
      genus_data.family,
      genus_data.magical_power
    );
    self->root = Optional(GenusNode).Of(root_node);
    self->magical_aura = genus_data.magical_power;
    
    // Add root to all_nodes as TreeNode choice type
    self->all_nodes.Push(TreeNode.Genus(root_node));
    
    // Create species branches
    var level: i32 = 1;
    for (species_name: String in genus_data.species) {
      var species_node: SpeciesNode = SpeciesNode.Create(
        species_name,
        self->genus_name,
        level,
        Random.Float64(0.5, 1.0)  // Random supernatural glow
      );
      self->species_nodes.Push(species_node);
      self->all_nodes.Push(TreeNode.Species(species_node));
      level = level + 1;
    }
    
    self->height = self->species_nodes.Size();
  }
  
  // Grow the tree by adding supernatural ornaments
  fn Grow[addr self: Self*](iterations: i32) {
    var i: i32 = 0;
    while (i < iterations) {
      self->AddSupernaturalOrnaments();
      self->EnhanceMagicalAura();
      i = i + 1;
    }
  }
  
  // Add supernatural ornaments to the tree
  fn AddSupernaturalOrnaments[addr self: Self*]() {
    var num_ornaments: i32 = Random.Int32(1, 3);
    
    var i: i32 = 0;
    while (i < num_ornaments) {
      // Random position
      var level: i32 = Random.Int32(1, self->height);
      var offset: i32 = Random.Int32(0, level);
      
      // Random color and effect
      var color: OrnamentColor = self->RandomColor();
      var effect: MagicalEffect = self->RandomEffect();
      
      // Calculate intensity based on position
      var intensity: f64 = (self->height - level + 1) as f64 / self->height as f64;
      
      var ornament: OrnamentNode = OrnamentNode.Create(
        color,
        effect,
        intensity,
        StringFormat("Beauty at level {0}", level),
        level,
        offset
      );
      
      self->ornaments.Push(ornament);
      self->all_nodes.Push(TreeNode.Ornament(ornament));
      
      i = i + 1;
    }
  }
  
  // Enhance the magical aura
  fn EnhanceMagicalAura[addr self: Self*]() {
    self->magical_aura = Min(1.0, self->magical_aura * 1.1);
  }
  
  // Helper to get random color
  fn RandomColor[self: Self]() -> OrnamentColor {
    var rand: i32 = Random.Int32(0, 6);
    if (rand == 0) { return OrnamentColor.Red; }
    if (rand == 1) { return OrnamentColor.Gold; }
    if (rand == 2) { return OrnamentColor.Blue; }
    if (rand == 3) { return OrnamentColor.Green; }
    if (rand == 4) { return OrnamentColor.Purple; }
    if (rand == 5) { return OrnamentColor.White; }
    return OrnamentColor.Silver;
  }
  
  // Helper to get random effect
  fn RandomEffect[self: Self]() -> MagicalEffect {
    var rand: i32 = Random.Int32(0, 5);
    if (rand == 0) { return MagicalEffect.Sparkle; }
    if (rand == 1) { return MagicalEffect.Star; }
    if (rand == 2) { return MagicalEffect.Snowflake; }
    if (rand == 3) { return MagicalEffect.Glow; }
    if (rand == 4) { return MagicalEffect.Shine; }
    return MagicalEffect.Twinkle;
  }
  
  // Filter nodes by type using pattern matching
  fn FilterByType[self: Self](node_type: TreeNodeType) -> Vector(TreeNode) {
    var filtered: Vector(TreeNode) = Vector(TreeNode).Create();
    
    for (node: TreeNode in self.all_nodes) {
      if (GetTreeNodeType(node) == node_type) {
        filtered.Push(node);
      }
    }
    
    return filtered;
  }
  
  // Get tree statistics
  fn GetStatistics[self: Self]() {
    Print("=== Supernatural Christmas Tree Statistics ===");
    Print("Genus: {0}", self.genus_name);
    
    if (self.root.HasValue()) {
      var root_node: GenusNode = self.root.Value();
      Print("Common Name: {0}", root_node.common_name);
      Print("Family: {0}", root_node.family);
    }
    
    Print("Height: {0} levels", self.height);
    Print("Number of Species: {0}", self.species_nodes.Size());
    Print("Number of Ornaments: {0}", self.ornaments.Size());
    Print("Magical Aura: {0}%", self.magical_aura * 100.0);
  }
  
  // Display tree structure with discriminated union info
  fn DisplayDiscriminatedUnion[self: Self]() {
    Print("\n╔═══════════════════════════════════════════════════════════════╗");
    Print("║    DISCRIMINATED UNION IN CARBON SUPERNATURAL CHRISTMAS TREE  ║");
    Print("╚═══════════════════════════════════════════════════════════════╝\n");
    
    Print("Carbon's choice types are discriminated unions (sum types) that");
    Print("can hold values of different types with compile-time type safety.\n");
    
    Print("TreeNode Choice Type Definition:");
    Print("choice TreeNode {");
    Print("  Genus(GenusNode),");
    Print("  Species(SpeciesNode),");
    Print("  Ornament(OrnamentNode)");
    Print("}\n");
    
    Print("Pattern Matching Example:");
    Print("match (node) {");
    Print("  case TreeNode.Genus(g) => { /* handle genus */ }");
    Print("  case TreeNode.Species(s) => { /* handle species */ }");
    Print("  case TreeNode.Ornament(o) => { /* handle ornament */ }");
    Print("}\n");
    
    // Display node type distribution
    Print("Node Type Distribution in This Tree:");
    Print("─────────────────────────────────────");
    
    var genus_count: i32 = self.FilterByType(TreeNodeType.Genus).Size();
    var species_count: i32 = self.FilterByType(TreeNodeType.Species).Size();
    var ornament_count: i32 = self.FilterByType(TreeNodeType.Ornament).Size();
    
    Print("GENUS nodes:    {0}", genus_count);
    Print("SPECIES nodes:  {0}", species_count);
    Print("ORNAMENT nodes: {0}", ornament_count);
    Print("TOTAL nodes:    {0}", self.all_nodes.Size());
    
    Print("\nThis demonstrates Carbon's type-safe choice types, similar to:");
    Print("  • Rust: enum with data");
    Print("  • TypeScript: discriminated unions");
    Print("  • F#/OCaml: discriminated unions");
    Print("  • Haskell: algebraic data types");
    Print("  • C++: std::variant (but with better ergonomics)\n");
  }
  
  // Visualize the tree in ASCII art
  fn VisualizeAscii[self: Self]() {
    Print("\n       ⭐\n");  // Star at top
    
    // Build tree from top to bottom
    var level: i32 = 1;
    while (level <= self.height) {
      var width: i32 = level * 2 + 1;
      var padding: i32 = self.height - level + 1;
      
      // Print padding spaces
      var p: i32 = 0;
      while (p < padding) {
        Print(" ", end="");
        p = p + 1;
      }
      
      // Print ornaments at this level (simplified)
      var ornament_count: i32 = 0;
      for (ornament: OrnamentNode in self.ornaments) {
        if (ornament.position_level == level and ornament_count < 2) {
          Print("✨", end="");
          ornament_count = ornament_count + 1;
        }
      }
      
      // Print tree branches
      var w: i32 = 0;
      while (w < width) {
        Print("🌲", end="");
        w = w + 1;
      }
      Print("");  // Newline
      
      level = level + 1;
    }
    
    // Print trunk
    var trunk_padding: i32 = self.height - 1;
    var t: i32 = 0;
    while (t < 2) {
      var p: i32 = 0;
      while (p < trunk_padding) {
        Print(" ", end="");
        p = p + 1;
      }
      Print("🟫🟫🟫");
      t = t + 1;
    }
    
    // Print magical aura bar
    var aura_stars: i32 = (self.magical_aura * 10.0) as i32;
    Print("\nMagical Aura: [", end="");
    var s: i32 = 0;
    while (s < aura_stars) {
      Print("✨", end="");
      s = s + 1;
    }
    Print("] {0}%\n", self.magical_aura * 100.0);
  }
}

// ============================================================================
// MAIN DEMO FUNCTION
// ============================================================================

fn Main() -> i32 {
  Print("🎄 SUPERNATURAL CHRISTMAS TREE GENERATOR (CARBON EDITION) 🎄");
  Print("═══════════════════════════════════════════════════════════════");
  Print("Demonstrating Carbon's Choice Types (Discriminated Unions)\n");
  
  // List available genera
  Print("Available Botanical Genera for Christmas Trees:");
  var genera: Vector(String) = BotanicalDatabase.ListGenera();
  for (genus: String in genera) {
    var data_opt: Optional(GenusData) = BotanicalDatabase.GetGenusData(genus);
    if (data_opt.HasValue()) {
      var data: GenusData = data_opt.Value();
      Print("  • {0} ({1}) - Magical Power: {2}", genus, data.common_name, data.magical_power);
    }
  }
  Print("");
  
  // Create a supernatural Christmas tree
  Print("Creating a supernatural Christmas tree from genus 'Abies' (Fir)...\n");
  var tree: SupernaturalChristmasTree = SupernaturalChristmasTree.Create("Abies");
  
  // Show initial statistics
  Print("Initial Tree Statistics:");
  tree.GetStatistics();
  Print("");
  
  // Grow the tree
  Print("Growing the tree with supernatural ornaments...\n");
  tree.Grow(3);
  
  // Show updated statistics
  Print("After Growth Statistics:");
  tree.GetStatistics();
  Print("");
  
  // Visualize the tree
  Print("ASCII Visualization:");
  tree.VisualizeAscii();
  
  // Explain discriminated unions
  tree.DisplayDiscriminatedUnion();
  
  // Demonstrate pattern matching with choice types
  Print("\n═══════════════════════════════════════════════════════════════");
  Print("PATTERN MATCHING EXAMPLE WITH CHOICE TYPES:");
  Print("═══════════════════════════════════════════════════════════════\n");
  
  Print("Displaying all nodes using pattern matching:\n");
  var count: i32 = 0;
  for (node: TreeNode in tree.all_nodes) {
    if (count < 8) {  // Show first 8 nodes
      DisplayNode(node);
      count = count + 1;
    }
  }
  
  if (tree.all_nodes.Size() > 8) {
    Print("... and {0} more nodes", tree.all_nodes.Size() - 8);
  }
  
  Print("\n═══════════════════════════════════════════════════════════════");
  Print("🎅 Merry Christmas from the Carbon Supernatural Tree! 🎅");
  Print("═══════════════════════════════════════════════════════════════");
  
  return 0;
}
