import asyncio

from alembic import context
from sqlalchemy.ext.asyncio import create_async_engine

from personal_assistant.config import Settings
from personal_assistant.models import Base


def run(connection):
    context.configure(
        connection=connection,
        target_metadata=Base.metadata,
        include_object=lambda obj, name, type_, reflected, compare_to: (
            not (type_ == "table" and name.startswith("checkpoint"))
        ),
    )
    with context.begin_transaction():
        context.run_migrations()


async def online():
    engine = create_async_engine(Settings().database_url)
    async with engine.connect() as connection:
        await connection.run_sync(run)
    await engine.dispose()


if context.is_offline_mode():
    context.configure(
        url=Settings().database_url, target_metadata=Base.metadata, literal_binds=True
    )
    with context.begin_transaction():
        context.run_migrations()
else:
    asyncio.run(online())
