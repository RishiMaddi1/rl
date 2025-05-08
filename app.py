import psycopg2
import random
import time
from collections import defaultdict

# === Database Setup ===

# Connect to default 'postgres' database first to create 'rldb'
def create_database():
    try:
        conn = psycopg2.connect(
            dbname='rl',
            user='rl',
            password='fO2doCgFHKKWaMiXNTj2WdKFr9iXX6YB',
            host='dpg-d0e3dg8dl3ps73baakk0-a',
            port='5432'
        )
        conn.autocommit = True
        cur = conn.cursor()

        # Create database 'rldb'
        cur.execute("SELECT 1 FROM pg_database WHERE datname = 'rldb'")
        exists = cur.fetchone()
        if not exists:
            cur.execute('CREATE DATABASE rldb')
            print("Database 'rldb' created.")
        else:
            print("Database 'rldb' already exists.")

        cur.close()
        conn.close()
    except Exception as e:
        print(f"Error creating database: {e}")

# Now connect to 'rldb'
def connect_to_db():
    try:
        return psycopg2.connect(
            dbname='rldb',
            user='rl',
            password='fO2doCgFHKKWaMiXNTj2WdKFr9iXX6YB',
            host='dpg-d0e3dg8dl3ps73baakk0-a',
            port='5432'
        )
    except Exception as e:
        print(f"Error connecting to database: {e}")
        return None

# Create tables
def create_tables(cur):
    try:
        cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
          id SERIAL PRIMARY KEY,
          name TEXT,
          age INT,
          city TEXT
        );
        """)

        cur.execute("""
        CREATE TABLE IF NOT EXISTS products (
          id SERIAL PRIMARY KEY,
          name TEXT,
          price NUMERIC,
          category TEXT
        );
        """)

        cur.execute("""
        CREATE TABLE IF NOT EXISTS orders (
          id SERIAL PRIMARY KEY,
          user_id INT REFERENCES users(id),
          product_id INT REFERENCES products(id),
          status TEXT,
          order_date DATE
        );
        """)

        print("Tables created.")
    except Exception as e:
        print(f"Error creating tables: {e}")

# Insert mock data
def insert_mock_data(cur):
    try:
        cur.execute("""
        INSERT INTO users (name, age, city)
        SELECT
          'User_' || i,
          (random() * 60 + 18)::int,
          CASE (random() * 5)::int
            WHEN 0 THEN 'New York'
            WHEN 1 THEN 'London'
            WHEN 2 THEN 'Tokyo'
            WHEN 3 THEN 'Delhi'
            ELSE 'Berlin'
          END
        FROM generate_series(1, 1000) AS s(i)
        ON CONFLICT DO NOTHING;
        """)

        cur.execute("""
        INSERT INTO products (name, price, category)
        SELECT
          'Product_' || i,
          round((random() * 500 + 20)::numeric, 2),
          CASE (random() * 3)::int
            WHEN 0 THEN 'Electronics'
            WHEN 1 THEN 'Books'
            ELSE 'Clothing'
          END
        FROM generate_series(1, 1000) AS s(i)
        ON CONFLICT DO NOTHING;
        """)

        cur.execute("""
        INSERT INTO orders (user_id, product_id, status, order_date)
        SELECT
          (random() * 999 + 1)::int,
          (random() * 999 + 1)::int,
          CASE (random() * 2)::int
            WHEN 0 THEN 'delivered'
            ELSE 'pending'
          END,
          NOW() - (random() * 365 || ' days')::interval
        FROM generate_series(1, 10000)
        ON CONFLICT DO NOTHING;
        """)

        print("Mock data inserted successfully.")
    except Exception as e:
        print(f"Error inserting mock data: {e}")

# === RL-based Cache Simulation ===

# Define Q-Learning Agent for cache management
class QLearningCache:
    def __init__(self, capacity=10, alpha=0.1, gamma=0.9, epsilon=0.1, decay_rate=0.75):
        self.capacity = capacity
        self.cache = {}  # Stores actual cache data
        self.q_table = defaultdict(lambda: defaultdict(float))  # Stores Q-values for each query
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.decay_rate = decay_rate  # Decay rate per step

    def decay_q_table(self):
        # Apply decay to all Q-values
        for state in self.q_table:
            for action in self.q_table[state]:
                self.q_table[state][action] *= self.decay_rate

    def get(self, key):
        # Exploration vs Exploitation
        if random.uniform(0, 1) < self.epsilon:
            action = random.choice(["evict", "cache"])
        else:
            action = "evict" if self.q_table[key]["evict"] > self.q_table[key]["cache"] else "cache"

        if action == "cache":
            return self.cache.get(key)
        return None

    def put(self, key, value):
        if len(self.cache) >= self.capacity:
            # Evict least valuable
            min_q_value_query = min(self.q_table, key=lambda k: self.q_table[k]["cache"])
            self.cache.pop(min_q_value_query, None)

        self.cache[key] = value
        # Update Q-values
        reward = 1
        old_q_value = self.q_table[key]["cache"]
        self.q_table[key]["cache"] = old_q_value + self.alpha * (reward + self.gamma * max(self.q_table[key].values()) - old_q_value)

    def update_q_value(self, key, action, reward):
        self.q_table[key][action] += reward

# Define queries
queries = [
    ("city_users", "SELECT * FROM users WHERE city = %s", lambda: random.choice(['New York', 'London', 'Tokyo', 'Delhi', 'Berlin'])),
    ("category_products", "SELECT * FROM products WHERE category = %s", lambda: random.choice(['Electronics', 'Books', 'Clothing'])),
    ("product_orders", "SELECT * FROM orders WHERE product_id = %s", lambda: random.randint(1, 10)),
]

# Simulating a real-world query distribution
def skewed_query_distribution():
    query_name = random.choice([q[0] for q in queries])

    if query_name == "city_users":
        city = random.choices(
            ['New York', 'Delhi', 'London', 'Tokyo', 'Berlin'],
            weights=[0.4, 0.3, 0.1, 0.1, 0.1],
            k=1
        )[0]
        return query_name, city
    elif query_name == "category_products":
        category = random.choices(
            ['Electronics', 'Books', 'Clothing'],
            weights=[0.4, 0.3, 0.3],
            k=1
        )[0]
        return query_name, category
    else:
        product_id = random.randint(1, 10)
        return query_name, product_id

# Initialize RL-based cache and stats
cache = QLearningCache(capacity=10)
query_stats = {q[0]: [] for q in queries}

total_queries = 10000

print("Running simulation with RL-based cache enabled...\n")

# === Simulation ===
def run_simulation_with_cache(cur):
    for i in range(total_queries):
        q_name, param = skewed_query_distribution()

        cache_key = (q_name, param)
        start = time.time()

        cached_result = cache.get(cache_key)
        if cached_result:
            result = cached_result
            reward = 1
        else:
            q_text = next(q[1] for q in queries if q[0] == q_name)
            cur.execute(q_text, (param,))
            result = cur.fetchall()
            cache.put(cache_key, result)
            reward = 1

        cache.update_q_value(cache_key, "cache" if cached_result else "evict", reward)
        cache.decay_q_table()

        elapsed_ms = (time.time() - start) * 1000
        query_stats[q_name].append(elapsed_ms)

        if i % 1000 == 0 and i > 0:
            print(f"{i} queries simulated...")

        time.sleep(0.001)

# === Simulate Without Caching ===
def run_simulation_without_cache(cur):
    no_cache_stats = {q[0]: [] for q in queries}

    for i in range(total_queries):
        q_name, param = skewed_query_distribution()

        start = time.time()
        q_text = next(q[1] for q in queries if q[0] == q_name)
        cur.execute(q_text, (param,))
        result = cur.fetchall()
        elapsed_ms = (time.time() - start) * 1000
        no_cache_stats[q_name].append(elapsed_ms)

        if i % 1000 == 0 and i > 0:
            print(f"{i} queries simulated (No cache)...")

        time.sleep(0.001)

    return no_cache_stats

# === Summary ===
def print_summary(label, stats):
    all_times = sum([sum(times) for times in stats.values()])
    avg_time = all_times / total_queries
    print(f"{label}")
    print(f"  Total queries run: {total_queries}")
    print(f"  Total time taken: {all_times:.2f} ms")
    print(f"  Average query time: {avg_time:.2f} ms\n")
    for q_name in stats:
        count = len(stats[q_name])
        avg_q_time = sum(stats[q_name]) / count
        print(f"  - {q_name}: {count} queries, avg time {avg_q_time:.2f} ms")
    return all_times, avg_time

# Main function to run the simulation
def main():
    create_database()
    conn = connect_to_db()
    if conn is None:
        return

    cur = conn.cursor()

    create_tables(cur)
    insert_mock_data(cur)
    conn.commit()

    run_simulation_with_cache(cur)
    no_cache_stats = run_simulation_without_cache(cur)

    # Print both summaries and capture totals
    cached_total_time, cached_avg_time = print_summary("With RL-based Cache", query_stats)
    no_cache_total_time, no_cache_avg_time = print_summary("Without Cache (Baseline)", no_cache_stats)

    # === Comparative Metrics ===
    print("\n--- Performance Comparison ---\n")

    time_saved = no_cache_total_time - cached_total_time
    percent_improvement = (time_saved / no_cache_total_time) * 100
    speedup_factor = no_cache_avg_time / cached_avg_time

    print(f"Total time saved: {time_saved:.2f} ms")
    print(f"Overall speedup factor: {speedup_factor:.2f}x faster")
    print(f"Overall performance improvement: {percent_improvement:.2f}% faster on average\n")

    # Detailed per-query improvement
    print("Per-query type improvement:")
    for q_name in query_stats:
        cached_avg = sum(query_stats[q_name]) / len(query_stats[q_name])
        no_cache_avg = sum(no_cache_stats[q_name]) / len(no_cache_stats[q_name])
        q_time_saved = no_cache_avg - cached_avg
        q_percent_improve = (q_time_saved / no_cache_avg) * 100
        q_speedup_factor = no_cache_avg / cached_avg

        print(f"  - {q_name}:")
        print(f"      Avg time without cache: {no_cache_avg:.2f} ms")
        print(f"      Avg time with cache: {cached_avg:.2f} ms")
        print(f"      Speedup factor: {q_speedup_factor:.2f}x")
        print(f"      Improvement: {q_percent_improve:.2f}%")

    cur.close()
    conn.close()

if __name__ == "__main__":
    main()
